from flask import Flask, request, jsonify, send_file # Added send_file
import cv2
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import tempfile
import logging
from io import BytesIO # Added BytesIO

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
# === MODEL AND SCALER PATHS ===
# Ensure these paths are correct relative to where you run the script
ISO_MODEL_PATH = "iso_classifier_model.h5"
ISO_SCALER_PATH = "iso_feature_scaler.pkl"
ISO_MAP_PATH = "iso_class_to_label.pkl"
SS_MODEL_PATH = "stable_shutter_model.h5"
SS_SCALER_PATH = "stable_shutter_scaler.pkl"

# === ISO MODEL LOADING (Using user-provided logic) ===
iso_model = None
iso_scaler = None
class_to_iso = None
iso_classes = None
try:
    if os.path.exists(ISO_MODEL_PATH):
        iso_model = tf.keras.models.load_model(ISO_MODEL_PATH)
        logger.info(f"ISO Model loaded from {ISO_MODEL_PATH}")
    else:
        logger.error(f"ISO Model file not found at {ISO_MODEL_PATH}")

    if os.path.exists(ISO_SCALER_PATH):
        iso_scaler = joblib.load(ISO_SCALER_PATH)
        logger.info(f"ISO Scaler loaded from {ISO_SCALER_PATH}")
    else:
        logger.error(f"ISO Scaler file not found at {ISO_SCALER_PATH}")

    if os.path.exists(ISO_MAP_PATH):
        class_to_iso = joblib.load(ISO_MAP_PATH)
        iso_classes = [class_to_iso[i] for i in range(len(class_to_iso))]
        logger.info(f"ISO Class Map loaded from {ISO_MAP_PATH}")
    else:
        logger.error(f"ISO Class Map file not found at {ISO_MAP_PATH}")

except Exception as e:
    logger.error(f"Error loading ISO model/scaler/map: {e}", exc_info=True)
    # Ensure variables are None if loading failed
    iso_model = None
    iso_scaler = None
    class_to_iso = None
    iso_classes = None

# === SHUTTER SPEED MODEL LOADING (Using user-provided logic) ===
ss_model = None
ss_scaler = None
try:
    if os.path.exists(SS_MODEL_PATH):
        # Use compile=False as in the provided logic
        ss_model = load_model(SS_MODEL_PATH, compile=False)
        logger.info(f"Shutter Speed Model loaded from {SS_MODEL_PATH}")
    else:
        logger.error(f"Shutter Speed Model file not found at {SS_MODEL_PATH}")

    if os.path.exists(SS_SCALER_PATH):
        ss_scaler = joblib.load(SS_SCALER_PATH)
        logger.info(f"Shutter Speed Scaler loaded from {SS_SCALER_PATH}")
    else:
        logger.error(f"Shutter Speed Scaler file not found at {SS_SCALER_PATH}")

except Exception as e:
    # Use logger instead of print
    logger.error(f"⚠️ Shutter speed model/scaler load failed: {e}", exc_info=True)
    ss_model = None
    ss_scaler = None

# === COMMON FEATURE FUNCTIONS (Using user-provided logic) ===
# Added basic error checking (e.g., for None image) and logging

def brightness(image):
    if image is None: return 0.0
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    except Exception as e:
        logger.error(f"Error in brightness: {e}", exc_info=True)
        return 0.0

def histogram_stats(image):
    if image is None: return 0.0, 0.0
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
        mean = np.mean(hist)
        var = np.var(hist)
        return (float(mean) if np.isfinite(mean) else 0.0,
                float(var) if np.isfinite(var) else 0.0)
    except Exception as e:
        logger.error(f"Error in histogram_stats: {e}", exc_info=True)
        return 0.0, 0.0

def edge_density(image):
    if image is None: return 0.0
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0.0
        return float(density) if np.isfinite(density) else 0.0
    except Exception as e:
        logger.error(f"Error in edge_density: {e}", exc_info=True)
        return 0.0

def perceptual_blur_metric(image, threshold=0.1):
    if image is None: return 0.0
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        magnitude = np.sqrt(dx**2 + dy**2) # Using user's formula
        coords = np.column_stack(np.where(edges > 0))
        edge_widths = [1.0 / magnitude[y, x] for (y, x) in coords if magnitude[y, x] > threshold]
        pbm_score = np.mean(edge_widths) if edge_widths else 0.0
        return float(pbm_score) if np.isfinite(pbm_score) else 0.0
    except Exception as e:
        logger.error(f"Error in perceptual_blur_metric: {e}", exc_info=True)
        return 0.0

def laplacian_variance(image):
    if image is None: return 0.0
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return float(lap_var) if np.isfinite(lap_var) else 0.0
    except Exception as e:
        logger.error(f"Error in laplacian_variance: {e}", exc_info=True)
        return 0.0

def tenengrad_score(image):
    if image is None: return 0.0
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
        ten_score = np.mean(gx**2 + gy**2) # Using user's formula
        return float(ten_score) if np.isfinite(ten_score) else 0.0
    except Exception as e:
        logger.error(f"Error in tenengrad_score: {e}", exc_info=True)
        return 0.0

# === ISO Biasing Logic (Using user-provided logic) ===
def apply_iso_bias(predicted_iso, confidence):
    # Using the exact function provided by the user
    if predicted_iso > 2500:
        biased_iso = 2500
    elif predicted_iso in [2000, 2500] and confidence < 0.90:
        biased_iso = 1000
    elif predicted_iso == 1600 and confidence < 0.85:
        biased_iso = 800
    elif predicted_iso == 1250 and confidence < 0.7:
        biased_iso = 640
    elif predicted_iso == 1000 and confidence < 0.7:
        biased_iso = 500
    elif predicted_iso == 800 and confidence < 0.65:
        biased_iso = 400
    elif predicted_iso == 640 and confidence < 0.65:
        biased_iso = 250
    elif predicted_iso == 500 and confidence < 0.6:
        biased_iso = 125
    else:
        biased_iso = predicted_iso
    
    logger.debug(f"ISO Biasing - Raw: {predicted_iso}, Confidence: {confidence:.4f}, Biased: {biased_iso}")
    return biased_iso

# === ISO PREDICTION (Using user-provided logic) ===
def predict_iso_from_image(image):
    # Check if models are loaded
    if iso_model is None or iso_scaler is None or class_to_iso is None:
        logger.error("ISO model/scaler/map not loaded. Cannot predict ISO.")
        return None, None, None
    if image is None:
        logger.error("predict_iso_from_image called with None image.")
        return None, None, None

    try:
        # Calculate features
        feat_brightness = brightness(image)
        feat_hist_mean, feat_hist_var = histogram_stats(image)
        feat_edge_density = edge_density(image)
        feat_pbm = perceptual_blur_metric(image)

        # Check for NaN/Inf in raw features
        raw_features = [feat_brightness, feat_hist_mean, feat_hist_var, feat_edge_density, feat_pbm]
        if not all(np.isfinite(f) for f in raw_features):
            logger.error(f"NaN or Inf detected in ISO raw features: {raw_features}")
            return None, None, None

        # Create feature vector and scale (using user's shape)
        feature_vector = np.array([raw_features], dtype=np.float32)
        logger.debug(f"ISO Raw Features: {feature_vector}")
        feature_scaled = iso_scaler.transform(feature_vector)
        logger.debug(f"ISO Scaled Features: {feature_scaled}")

        # Predict probabilities
        pred_prob = iso_model.predict(feature_scaled)[0]
        logger.debug(f"ISO Prediction Probabilities: {pred_prob}")

        # Get predicted class, confidence, and raw ISO
        pred_class = np.argmax(pred_prob)
        confidence = pred_prob[pred_class]
        raw_iso = class_to_iso[pred_class]

        # Apply bias
        biased_iso = apply_iso_bias(raw_iso, confidence)

        logger.info(f"ISO Prediction - Raw: {raw_iso}, Confidence: {confidence:.4f}, Biased: {biased_iso}")
        return biased_iso, confidence, raw_iso # Return all three as per original function signature

    except Exception as e:
        logger.error(f"Error during ISO prediction: {e}", exc_info=True)
        return None, None, None

# === SHUTTER SPEED PREDICTION (Using user-provided logic) ===
def predict_shutter_speed_from_image(image):
    # Check if models are loaded
    if ss_model is None or ss_scaler is None:
        logger.error("Shutter speed model/scaler not loaded. Cannot predict shutter speed.")
        return None
    if image is None:
        logger.error("predict_shutter_speed_from_image called with None image.")
        return None

    try:
        # Calculate features
        lap = laplacian_variance(image)
        ten = tenengrad_score(image)
        pbm = perceptual_blur_metric(image)
        edge = edge_density(image)
        bright = brightness(image)
        hist_mean, hist_var = histogram_stats(image)

        # Check for NaN/Inf in raw features
        raw_features = [lap, ten, pbm, edge, bright, hist_mean, hist_var]
        if not all(np.isfinite(f) for f in raw_features):
            logger.error(f"NaN or Inf detected in SS raw features: {raw_features}")
            return None

        # Create feature vector and scale (using user's shape)
        feature_vector = np.array([raw_features], dtype=np.float32)
        logger.debug(f"SS Raw Features: {feature_vector}")
        scaled_input = ss_scaler.transform(feature_vector)
        logger.debug(f"SS Scaled Features: {scaled_input}")

        # Predict log shutter speed
        # Assuming model output is log(shutter + 1) based on np.expm1 usage in user's code
        log_shutter_plus_1 = ss_model.predict(scaled_input)[0][0]
        logger.debug(f"SS Raw Model Output (log_shutter+1?): {log_shutter_plus_1}")

        # Check for NaN/Inf in prediction
        if not np.isfinite(log_shutter_plus_1):
            logger.error(f"NaN or Inf predicted by SS model: {log_shutter_plus_1}")
            return None

        # Inverse transform: shutter = exp(log_value) - 1 (as per user's np.expm1)
        shutter = 2/ np.exp(log_shutter_plus_1)
        logger.debug(f"SS Calculated Shutter (before clip): {shutter:.6f}")

        # Clip the result to a practical range (1/1,000,000s to 60s) - as per user's code
        clipped_shutter = max(1e-6, min(shutter, 10.0))

        logger.info(f"Shutter Speed Prediction - Calculated: {shutter:.6f}, Clipped: {clipped_shutter:.6f}")
        return clipped_shutter

    except Exception as e:
        logger.error(f"Error during Shutter Speed prediction: {e}", exc_info=True)
        return None

# === Flask Endpoint (/recommend_settings - Adapted) ===
@app.route('/recommend_settings', methods=['POST'])
def recommend_settings():
    """
    Recommends camera settings (ISO, Shutter Speed) based on an uploaded image.
    POST multipart/form-data with field 'image'.
    Returns JSON: {'iso': recommended_iso, 'shutter_speed': recommended_shutter_speed}
    """
    if 'image' not in request.files:
        logger.warning("/recommend_settings: No image file found in request.")
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    img_file_path = None # Initialize path variable

    # Use tempfile for secure temporary file handling
    try:
        # Create a temporary file to save the uploaded image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            img_file_path = tmp.name
            img_file.save(img_file_path)
            logger.info(f"/recommend_settings: Image saved temporarily to {img_file_path}")

        # Read the image using OpenCV
        image = cv2.imread(img_file_path)

        if image is None:
            logger.error(f"/recommend_settings: Failed to read image file from {img_file_path}")
            return jsonify({'error': 'Invalid or corrupted image file'}), 400

        # --- Predict ISO using the user's function ---
        logger.info("Predicting ISO...")
        # We only need the final biased ISO for the response
        predicted_iso_biased, _, _ = predict_iso_from_image(image)
        if predicted_iso_biased is None:
             # Error already logged in predict function
             return jsonify({'error': 'ISO prediction failed'}), 500

        # --- Predict Shutter Speed using the user's function ---
        logger.info("Predicting Shutter Speed...")
        predicted_shutter_clipped = predict_shutter_speed_from_image(image)
        if predicted_shutter_clipped is None:
             # Error already logged in predict function
             return jsonify({'error': 'Shutter speed prediction failed'}), 500

        logger.info(f"/recommend_settings: Prediction results - ISO: {predicted_iso_biased}, Shutter: {predicted_shutter_clipped:.6f}")

        # --- Return JSON Response ---
        return jsonify({
            'iso': int(predicted_iso_biased), # Return biased ISO as integer
            'shutter_speed': float(predicted_shutter_clipped) # Return clipped shutter speed as float
        })

    except Exception as e:
        logger.error(f"Unexpected error in /recommend_settings: {e}", exc_info=True)
        return jsonify({'error': 'Internal server error during recommendation'}), 500
    finally:
        # Ensure the temp file is deleted even if errors occur
        if img_file_path and os.path.exists(img_file_path):
            try:
                os.unlink(img_file_path)
                logger.info(f"/recommend_settings: Temporary file {img_file_path} deleted.")
            except Exception as e:
                logger.error(f"Error deleting temp file {img_file_path}: {e}")


# === HEATMAP GENERATION LOGIC (Kept as is from previous version) ===
def generate_heatmap_overlay(image):
    """Generates a heatmap overlay based on vertical streaks."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        SHIFT = 30  # ≈ streak length in pixels
        # Ensure image height is sufficient for the shift
        if gray.shape[0] <= SHIFT:
            logger.warning(f"Image height ({gray.shape[0]}) is too small for shift ({SHIFT}). Heatmap might be inaccurate.")
            # Handle gracefully: maybe return original image or a blank heatmap
            # For now, let's proceed but the result might not be meaningful
            shifted = gray # No shift if too small
        else:
            shifted = np.roll(gray, -SHIFT, axis=0)  # vertical shift
            # Zero out the wrapped-around part at the bottom
            shifted[gray.shape[0]-SHIFT:, :] = gray[gray.shape[0]-SHIFT:, :]

        diff = cv2.absdiff(gray, shifted)  # |I - I_shift|
        inv = cv2.bitwise_not(diff)  # low diff -> bright
        blur = cv2.GaussianBlur(inv, (11, 11), 0)  # smooth glow
        norm = cv2.normalize(blur, None, 0, 255, cv2.NORM_MINMAX)

        heat = cv2.applyColorMap(norm.astype('uint8'), cv2.COLORMAP_TURBO)
        overlay = cv2.addWeighted(image, 0.6, heat, 0.8, 0)
        logger.info("Heatmap overlay generated successfully.")
        return overlay

    except Exception as e:
        logger.error(f"Error during heatmap generation: {e}", exc_info=True)
        return None

# === HEATMAP ENDPOINT (Kept as is from previous version) ===
@app.route('/generate_heatmap', methods=['POST'])
def generate_heatmap_endpoint():
    """
    POST /generate_heatmap
    - multipart/form-data with field 'image'
    - returns the generated heatmap overlay image as JPEG
    """
    if 'image' not in request.files:
        logger.warning("Heatmap request received without 'image' field.")
        return jsonify({'error': 'No image provided'}), 400

    img_file = request.files['image']
    img_file_path = None

    try:
        # Use tempfile for secure temporary file handling
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            img_file_path = tmp.name
            img_file.save(img_file_path)
            logger.info(f"Heatmap input image saved temporarily to {img_file_path}")

        # Read the image using OpenCV
        image = cv2.imread(img_file_path)

        if image is None:
            logger.error(f"Failed to read image file for heatmap: {img_file_path}")
            return jsonify({'error': 'Invalid image file'}), 400

        # --- Generate Heatmap ---
        heatmap_overlay = generate_heatmap_overlay(image)

        if heatmap_overlay is None:
            logger.error("Heatmap generation failed.")
            return jsonify({'error': 'Heatmap generation failed'}), 500

        # --- Encode result image to JPEG bytes ---
        is_success, buffer = cv2.imencode(".jpg", heatmap_overlay)
        if not is_success:
             logger.error("Failed to encode heatmap overlay to JPEG.")
             return jsonify({'error': 'Failed to encode result image'}), 500

        # --- Return image file ---
        logger.info("/generate_heatmap: Sending back processed heatmap image.")
        return send_file(
            BytesIO(buffer),
            mimetype='image/jpeg',
            as_attachment=False # Send inline
        )

    except Exception as e:
        logger.error(f"Error processing image in /generate_heatmap: {e}", exc_info=True)
        return jsonify({'error': 'Failed to process image for heatmap'}), 500
    finally:
        # Ensure the temp file is deleted
        if img_file_path and os.path.exists(img_file_path):
            try:
                os.unlink(img_file_path)
                logger.info(f"Temporary file {img_file_path} deleted.")
            except Exception as e:
                logger.error(f"Error deleting temp file {img_file_path}: {e}")

# === Main Execution (Adapted) ===
if __name__ == "__main__":
    # Check if all essential models/scalers for /recommend_settings loaded before starting
    recommend_models_loaded = iso_model and iso_scaler and class_to_iso and ss_model and ss_scaler

    if recommend_models_loaded:
        logger.info("All models for /recommend_settings loaded. Starting Flask server with all endpoints...")
    else:
        logger.critical("One or more ML models/scalers failed to load. /recommend_settings endpoint WILL NOT WORK.")
        logger.info("Starting Flask server ONLY for heatmap generation (/generate_heatmap)...")

    # Run the server regardless, but the warning indicates which endpoint might fail
    # Run on port 5001, accessible on local network
    # Set debug=False for production/stable testing
    app.run(host='0.0.0.0', port=5001, debug=False)