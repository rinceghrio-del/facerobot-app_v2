#include <jni.h>
#include <string>
#include <android/log.h>
#include "llama.h"
#include "common.h"

#define LOG_TAG "LlamaBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_facerobot_LlamaBridge_loadModel(JNIEnv* env, jobject, jstring modelPath) {
    const char* path = env->GetStringUTFChars(modelPath, nullptr);
    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    g_model = llama_load_model_from_file(path, mparams);
    env->ReleaseStringUTFChars(modelPath, path);
    if (!g_model) { LOGI("Failed to load model"); return JNI_FALSE; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_threads = 4;
    g_ctx = llama_new_context_with_model(g_model, cparams);
    if (!g_ctx) { LOGI("Failed to create context"); return JNI_FALSE; }

    LOGI("Model loaded successfully");
    return JNI_TRUE;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_facerobot_LlamaBridge_generate(JNIEnv* env, jobject, jstring prompt) {
    const char* p = env->GetStringUTFChars(prompt, nullptr);
    std::string fullPrompt = std::string("<|user|>\n") + p + "\n<|assistant|>\n";
    env->ReleaseStringUTFChars(prompt, p);

    std::vector<llama_token> tokens = common_tokenize(g_ctx, fullPrompt, true);
    llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
    if (llama_decode(g_ctx, batch) != 0) return env->NewStringUTF("");

    std::string result;
    int n_cur = (int)tokens.size();
    for (int i = 0; i < 200; i++) {
        auto* logits = llama_get_logits_ith(g_ctx, batch.n_tokens - 1);
        int n_vocab = llama_n_vocab(llama_get_model(g_ctx));
        llama_token new_token = 0;
        float best = logits[0];
        for (int t = 1; t < n_vocab; t++) if (logits[t] > best) { best = logits[t]; new_token = t; }

        if (new_token == llama_token_eos(g_model)) break;
        char buf[128];
        int n = llama_token_to_piece(g_model, new_token, buf, sizeof(buf), 0, true);
        result.append(buf, n);

        llama_token tok = new_token;
        batch = llama_batch_get_one(&tok, 1);
        if (llama_decode(g_ctx, batch) != 0) break;
        n_cur++;
    }
    return env->NewStringUTF(result.c_str());
}
