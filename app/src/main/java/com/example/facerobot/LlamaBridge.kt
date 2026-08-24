package com.example.facerobot

object LlamaBridge {
    init { System.loadLibrary("facerobot_llama") }
    external fun loadModel(modelPath: String): Boolean
    external fun generate(prompt: String): String
}
