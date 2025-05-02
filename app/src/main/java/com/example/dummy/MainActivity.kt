package com.example.dummy

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.*
import android.media.Image
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.material.switchmaterial.SwitchMaterial
import okhttp3.* // Import OkHttp classes
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject // Import JSONObject
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    private val TAG = "CameraAI_API"
    private val REQUEST_CODE_PERMISSIONS = 10
    private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)

    // --- Backend Server URL ---
    // Replace with your actual server IP (e.g., "http://192.168.1.100:5000/analyze")
    // Or use ngrok URL for testing
    private val SERVER_URL = "http://<your-ip-address>:5000/analyze"
    // --------------------------

    // UI Components
    private lateinit var previewView: PreviewView
    private lateinit var switchMode: SwitchMaterial
    private lateinit var buttonRecommend: Button
    private lateinit var buttonApplyAI: Button
    private lateinit var buttonCapture: Button
    private lateinit var imageManual: ImageView
    private lateinit var imageAI: ImageView
    private lateinit var comparisonLayout: LinearLayout

    // Camera variables
    private var camera: androidx.camera.core.Camera? = null
    private var imageCapture: ImageCapture? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var analysisExecutor: ExecutorService // Use for network calls too

    // OkHttp Client
    private val httpClient = OkHttpClient()

    // AI recommendation values
    private var recIso: Int = 0
    private var recShutter: Double = 0.0 // Changed to Double for API response

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Initialize UI components
        previewView = findViewById(R.id.previewView)
        switchMode = findViewById(R.id.switchMode)
        buttonRecommend = findViewById(R.id.buttonRecommend)
        buttonApplyAI = findViewById(R.id.buttonApplyAI)
        buttonCapture = findViewById(R.id.buttonCapture)
        imageManual = findViewById(R.id.imageManual)
        imageAI = findViewById(R.id.imageAI)
        comparisonLayout = findViewById(R.id.comparisonLayout)

        // Initialize executors
        cameraExecutor = Executors.newSingleThreadExecutor()
        analysisExecutor = Executors.newSingleThreadExecutor() // Re-use for network

        // Check camera permissions
        if (allPermissionsGranted()) {
            startCamera()
        } else {
            ActivityCompat.requestPermissions(
                this, REQUIRED_PERMISSIONS, REQUEST_CODE_PERMISSIONS
            )
        }

        // Set up UI listeners
        setupUIListeners()
    }

    private fun setupUIListeners() {
        // Mode switch listener
        switchMode.setOnCheckedChangeListener { _, isChecked ->
            val mode = if (isChecked) "AI Mode ON" else "Manual Mode ON"
            Toast.makeText(this, mode, Toast.LENGTH_SHORT).show()
            if (!isChecked) comparisonLayout.visibility = View.GONE
        }

        // Recommend button listener
        buttonRecommend.setOnClickListener {
            if (switchMode.isChecked) {
                analyzeCurrentFrameViaAPI() // Call the API version
            } else {
                Toast.makeText(this, "Enable AI Mode to get recommendations", Toast.LENGTH_SHORT).show()
            }
        }

        // Apply AI settings button listener
        buttonApplyAI.setOnClickListener {
            if (camera != null && switchMode.isChecked) {
                applyAISettings()
            } else {
                Toast.makeText(this, "Enable AI Mode to apply AI settings", Toast.LENGTH_SHORT).show()
            }
        }

        // Capture button listener
        buttonCapture.setOnClickListener {
            takePhoto()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(previewView.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider?.unbindAll()
                camera = cameraProvider?.bindToLifecycle(
                    this, cameraSelector, preview, imageCapture
                )
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    // Renamed function to reflect API usage
    private fun analyzeCurrentFrameViaAPI() {
        val imageCapture = imageCapture ?: return
        Toast.makeText(this, "Analyzing image via API...", Toast.LENGTH_SHORT).show()

        imageCapture.takePicture(
            analysisExecutor, // Use background thread
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    // Convert ImageProxy to byte array (JPEG)
                    val bytes = imageProxyToJpegByteArray(imageProxy) // Use JPEG conversion
                    imageProxy.close()

                    if (bytes != null) {
                        // Send bytes to the backend API
                        sendImageToAPI(bytes)
                    } else {
                        runOnUiThread { Toast.makeText(this@MainActivity, "Failed to convert image", Toast.LENGTH_SHORT).show() }
                    }
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Image capture for analysis failed: ${exception.message}", exception)
                    runOnUiThread { Toast.makeText(this@MainActivity, "Capture failed: ${exception.message}", Toast.LENGTH_SHORT).show() }
                }
            }
        )
    }

    // Function to send image bytes to the backend API
    private fun sendImageToAPI(imageBytes: ByteArray) {
        Log.d(TAG, "Sending image to: $SERVER_URL")
        val requestBody = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart(
                "image", // This key must match the backend (Flask/FastAPI)
                "photo.jpg",
                imageBytes.toRequestBody("image/jpeg".toMediaTypeOrNull(), 0, imageBytes.size)
            )
            .build()

        val request = Request.Builder()
            .url(SERVER_URL) // Use the defined server URL
            .post(requestBody)
            .build()

        httpClient.newCall(request).enqueue(object : Callback {
            override fun onFailure(call: Call, e: IOException) {
                Log.e(TAG, "API call failed: ${e.message}", e)
                runOnUiThread {
                    Toast.makeText(this@MainActivity, "API Error: ${e.message}", Toast.LENGTH_LONG).show()
                }
            }

            override fun onResponse(call: Call, response: Response) {
                if (!response.isSuccessful) {
                    Log.e(TAG, "API Error: ${response.code} ${response.message}")
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "API Error: ${response.code}", Toast.LENGTH_SHORT).show()
                    }
                    response.close() // Close the response body
                    return
                }

                val responseBody = response.body?.string()
                response.close() // Close the response body

                try {
                    val json = JSONObject(responseBody ?: "{}")
                    val blurScore = json.optDouble("blur_score", 0.0)
                    val brightness = json.optDouble("brightness", 0.0)
                    val rec = json.optJSONObject("recommendations")

                    if (rec != null) {
                        recIso = rec.optInt("iso", 100)
                        recShutter = rec.optDouble("shutter", 1.0/125.0) // Default shutter

                        // Update UI with recommendations
                        runOnUiThread {
                            val message = "API recommends:\n" +
                                    "- ISO: $recIso\n" +
                                    "- Shutter: 1/${(1 / recShutter).toInt()}s\n" + // Display as fraction
                                    "- Blur Score: %.1f\n".format(blurScore) +
                                    "- Brightness: %.1f".format(brightness)

                            Toast.makeText(this@MainActivity, message, Toast.LENGTH_LONG).show()
                        }
                    } else {
                         runOnUiThread { Toast.makeText(this@MainActivity, "Invalid API response format", Toast.LENGTH_SHORT).show() }
                    }

                } catch (e: Exception) {
                    Log.e(TAG, "Failed to parse API response: ${e.message}", e)
                    runOnUiThread {
                        Toast.makeText(this@MainActivity, "API Response Parse Error", Toast.LENGTH_SHORT).show()
                    }
                }
            }
        })
    }


    private fun applyAISettings() {
        // This function remains largely the same, using recIso, recShutter (if applicable)
        // For simplicity, we still use Exposure Compensation based on brightness/recommendation
        camera?.let { cam ->
            val exposureState = cam.cameraInfo.exposureState
            if (exposureState.isExposureCompensationSupported) {
                // Example: Map ISO/Shutter recommendation to exposure index (needs refinement)
                var exposureIndex = 0
                if (recIso > 400 || recShutter > 1.0/60.0) { // Simple logic: if high ISO or slow shutter needed -> brighter scene needed
                    exposureIndex = 2 // Increase exposure
                } else if (recIso < 200 && recShutter < 1.0/250.0) { // Low ISO, fast shutter -> darker scene needed
                    exposureIndex = -2 // Decrease exposure
                }

                val indexRange = exposureState.exposureCompensationRange
                val safeIndex = exposureIndex.coerceIn(indexRange.lower, indexRange.upper)

                cam.cameraControl.setExposureCompensationIndex(safeIndex)
                    .addListener({
                        Toast.makeText(this, "Applied exposure compensation: $safeIndex", Toast.LENGTH_SHORT).show()
                    }, ContextCompat.getMainExecutor(this))
            } else {
                Toast.makeText(this, "Exposure compensation not supported", Toast.LENGTH_SHORT).show()
            }
        }
    }

    private fun takePhoto() {
        buttonCapture.isEnabled = false
        val imageCapture = imageCapture ?: return

        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onCaptureSuccess(imageProxy: ImageProxy) {
                    val bitmap = imageProxyToBitmap(imageProxy) // Use existing conversion
                    imageProxy.close()

                    if (bitmap != null) {
                        if (switchMode.isChecked) {
                            imageAI.setImageBitmap(bitmap)
                            comparisonLayout.visibility = View.VISIBLE
                            // You might capture another image with default settings for imageManual
                            imageManual.setImageBitmap(bitmap) // Placeholder
                        } else {
                            imageManual.setImageBitmap(bitmap)
                            Toast.makeText(this@MainActivity, "Image captured!", Toast.LENGTH_SHORT).show()
                        }
                    }
                    buttonCapture.isEnabled = true
                }

                override fun onError(exception: ImageCaptureException) {
                    Log.e(TAG, "Photo capture failed: ${exception.message}", exception)
                    Toast.makeText(this@MainActivity, "Capture failed: ${exception.message}", Toast.LENGTH_SHORT).show()
                    buttonCapture.isEnabled = true
                }
            }
        )
    }

    // Convert ImageProxy to JPEG byte array (More efficient than YUV->Bitmap->JPEG)
    private fun imageProxyToJpegByteArray(image: ImageProxy): ByteArray? {
        // Check if the format is JPEG directly
        if (image.format == ImageFormat.JPEG) {
            val buffer = image.planes[0].buffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            return bytes
        }
        // If not JPEG, convert YUV_420_888 to JPEG
        else if (image.format == ImageFormat.YUV_420_888) {
            val yBuffer = image.planes[0].buffer
            val uBuffer = image.planes[1].buffer
            val vBuffer = image.planes[2].buffer

            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()

            val nv21 = ByteArray(ySize + uSize + vSize)

            // Get the Y plane
            yBuffer.get(nv21, 0, ySize)
            // Get the U and V planes (interleaved in NV21 format: V first then U)
            vBuffer.get(nv21, ySize, vSize)
            // Correctly copy U plane after V plane, considering pixel/row strides if necessary
            // This simple copy assumes contiguous planes, which might not always be true.
            // A more robust solution might involve checking strides.
            val uPlaneBytes = ByteArray(uSize)
            uBuffer.get(uPlaneBytes)
            // Interleave U bytes into the NV21 buffer after V bytes
            var uIndex = 0
            for (i in ySize + vSize until nv21.size) {
                 if (i % 2 == 0 && uIndex < uPlaneBytes.size) { // Place U byte at even indices after V
                     nv21[i] = uPlaneBytes[uIndex++]
                 }
            }


            try {
                val yuvImage = YuvImage(nv21, ImageFormat.NV21, image.width, image.height, null)
                val out = ByteArrayOutputStream()
                // Compress YUV to JPEG
                yuvImage.compressToJpeg(Rect(0, 0, image.width, image.height), 90, out) // Quality 90
                return out.toByteArray()
            } catch (e: Exception) {
                Log.e(TAG, "YUV to JPEG conversion failed: ${e.message}")
                return null
            }
        } else {
            Log.e(TAG, "Unsupported image format for JPEG conversion: ${image.format}")
            return null // Or attempt Bitmap conversion as fallback
        }
    }


    // Utility: convert ImageProxy to Bitmap (Keep as fallback or for display)
    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        val jpegBytes = imageProxyToJpegByteArray(imageProxy) // Reuse JPEG conversion
        return if (jpegBytes != null) {
            val bitmap = BitmapFactory.decodeByteArray(jpegBytes, 0, jpegBytes.size)
            // Rotate if necessary (though JPEG often includes orientation)
             if (bitmap != null && imageProxy.imageInfo.rotationDegrees != 0) {
                 val matrix = Matrix()
                 matrix.postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
                 Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
             } else {
                 bitmap
             }
        } else {
            null // Failed to get JPEG bytes
        }
    }


    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (allPermissionsGranted()) {
                startCamera()
            } else {
                Toast.makeText(this, "Permissions not granted.", Toast.LENGTH_SHORT).show()
                finish()
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        analysisExecutor.shutdown()
        // Cancel OkHttp calls if needed
        httpClient.dispatcher.executorService.shutdown()
        httpClient.connectionPool.evictAll()
    }
}