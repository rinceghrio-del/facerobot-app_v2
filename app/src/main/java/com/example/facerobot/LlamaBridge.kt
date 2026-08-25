package com.example.facerobot

object LlamaBridge {
    init { System.loadLibrary("facerobot_llama") }
    external fun loadModel(modelPath: String): Boolean
    external fun generate(prompt: String): String

    val logEntries = mutableListOf<Pair<Long, String>>()
    private const val MAX_LOG = 200

    @JvmStatic
    fun appendLog(msg: String) {
        synchronized(logEntries) {
            logEntries.add(0, System.currentTimeMillis() to msg)
            if (logEntries.size > MAX_LOG) logEntries.removeAt(logEntries.size - 1)
        }
    }
}
