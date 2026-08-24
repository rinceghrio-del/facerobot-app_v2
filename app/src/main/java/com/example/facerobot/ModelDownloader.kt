package com.example.facerobot

import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL

object ModelDownloader {
    private const val MODEL_URL =
        "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    private const val MODEL_FILENAME = "llama-3.2-1b-instruct-q4.gguf"

    fun getModelFile(context: Context): File = File(context.filesDir, MODEL_FILENAME)

    fun isModelDownloaded(context: Context): Boolean {
        val f = getModelFile(context)
        return f.exists() && f.length() > 500_000_000L // sanity check, dapat malaki (~800MB)
    }

    // Tawagin ito sa background thread (hindi sa main/UI thread)
    fun downloadModel(context: Context, onProgress: (Int) -> Unit, onDone: (Boolean) -> Unit) {
        val outFile = getModelFile(context)
        try {
            val url = URL(MODEL_URL)
            val conn = url.openConnection() as HttpURLConnection
            conn.connect()
            val totalSize = conn.contentLength
            var downloaded = 0

            conn.inputStream.use { input ->
                FileOutputStream(outFile).use { output ->
                    val buffer = ByteArray(8192)
                    var bytesRead: Int
                    while (input.read(buffer).also { bytesRead = it } != -1) {
                        output.write(buffer, 0, bytesRead)
                        downloaded += bytesRead
                        if (totalSize > 0) {
                            onProgress((downloaded * 100L / totalSize).toInt())
                        }
                    }
                }
            }
            onDone(true)
        } catch (e: Exception) {
            Log.e("ModelDownloader", "Download failed: ${e.message}")
            outFile.delete()
            onDone(false)
        }
    }
}
