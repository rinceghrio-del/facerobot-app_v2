#include <jni.h>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <android/log.h>
#include "llama.h"
#include "common.h"

#define LOG_TAG "LlamaBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;

// Nagpapadala ng log message papunta sa LlamaBridge.appendLog() sa Kotlin side,
// para makita ito sa "Llama Log" sa loob ng app menu, kasabay ng normal Logcat print.
static void jlog(JNIEnv* env, const char* fmt, ...) {
    char buf[256];
    va_list args;
    va_start(args, fmt);
    vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);

    LOGI("%s", buf);
    jclass cls = env->FindClass("com/example/facerobot/LlamaBridge");
    if (cls) {
        jmethodID mid = env->GetStaticMethodID(cls, "appendLog", "(Ljava/lang/String;)V");
        if (mid) {
            jstring jmsg = env->NewStringUTF(buf);
            env->CallStaticVoidMethod(cls, mid, jmsg);
            env->DeleteLocalRef(jmsg);
        }
        env->DeleteLocalRef(cls);
    }
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_example_facerobot_LlamaBridge_loadModel(JNIEnv* env, jobject, jstring modelPath) {
    try {
        const char* path = env->GetStringUTFChars(modelPath, nullptr);
        llama_backend_init();

        llama_model_params mparams = llama_model_default_params();
        g_model = llama_model_load_from_file(path, mparams);
        env->ReleaseStringUTFChars(modelPath, path);
        if (!g_model) { jlog(env, "Failed to load model"); return JNI_FALSE; }

        g_vocab = llama_model_get_vocab(g_model);

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = 512;      // pinaliit para sa mas mababang RAM usage
        cparams.n_threads = 2;
        cparams.n_threads_batch = 2;
        g_ctx = llama_init_from_model(g_model, cparams);
        if (!g_ctx) { jlog(env, "Failed to create context"); return JNI_FALSE; }

        jlog(env, "Model loaded successfully");
        return JNI_TRUE;
    } catch (const std::exception& e) {
        jlog(env, "Exception in loadModel: %s", e.what());
        return JNI_FALSE;
    } catch (...) {
        jlog(env, "Unknown exception in loadModel (likely out of memory)");
        return JNI_FALSE;
    }
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_facerobot_LlamaBridge_generate(JNIEnv* env, jobject, jstring prompt) {
    try {
        if (!g_ctx || !g_model || !g_vocab) {
            jlog(env, "generate() called but model not loaded");
            return env->NewStringUTF("");
        }

        const char* p = env->GetStringUTFChars(prompt, nullptr);
        std::string fullPrompt =
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "Ikaw ay si Rustech, isang helpful robot assistant. Sumagot nang maikli (1-2 pangungusap) sa Taglish.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + std::string(p) + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
        env->ReleaseStringUTFChars(prompt, p);

        std::vector<llama_token> tokens = common_tokenize(g_ctx, fullPrompt, true, true);
        llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
        if (llama_decode(g_ctx, batch) != 0) {
            jlog(env, "Initial decode failed");
            return env->NewStringUTF("");
        }

        std::string result;
        int n_vocab = llama_vocab_n_tokens(g_vocab);
        for (int i = 0; i < 80; i++) {
            auto* logits = llama_get_logits_ith(g_ctx, batch.n_tokens - 1);
            if (!logits) { jlog(env, "Null logits at step %d", i); break; }

            llama_token new_token = 0;
            float best = logits[0];
            for (int t = 1; t < n_vocab; t++) if (logits[t] > best) { best = logits[t]; new_token = t; }

            if (llama_vocab_is_eog(g_vocab, new_token)) break;
            char buf[128];
            int n = llama_token_to_piece(g_vocab, new_token, buf, sizeof(buf), 0, true);
            if (n > 0) result.append(buf, n);

            llama_token tok = new_token;
            batch = llama_batch_get_one(&tok, 1);
            if (llama_decode(g_ctx, batch) != 0) {
                jlog(env, "Decode failed at step %d", i);
                break;
            }
        }
        jlog(env, "Llama reply (%d chars): %s", (int)result.size(), result.c_str());
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        jlog(env, "Exception in generate: %s", e.what());
        return env->NewStringUTF("");
    } catch (...) {
        jlog(env, "Unknown exception in generate (likely out of memory)");
        return env->NewStringUTF("");
    }
}
