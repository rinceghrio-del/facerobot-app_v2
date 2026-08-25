#include <jni.h>
#include <string>
#include <cstdio>
#include <cstdarg>
#include <mutex>
#include <thread>
#include <algorithm>
#include <android/log.h>
#include "llama.h"
#include "common.h"

#define LOG_TAG "LlamaBridge"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

static llama_model* g_model = nullptr;
static llama_context* g_ctx = nullptr;
static const llama_vocab* g_vocab = nullptr;
static int g_n_ctx = 0;

// Pinoprotektahan ng mutex na ito ang g_model/g_ctx. Kahit may guard na sa Kotlin
// side (llamaBusy flag) laban sa sabay-sabay na tawag, panatilihin din ang lock dito
// bilang huling proteksyon - ang llama_context ay HINDI thread-safe, at kapag dalawang
// thread ang sabay na tumawag ng llama_decode() dito, native crash (segfault) ang
// resulta na hindi mahuhuli ng try/catch, kahit sa C++.
static std::mutex g_mutex;

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
    std::lock_guard<std::mutex> lock(g_mutex);
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
        // Dating naka-hardcode sa 2 threads lang kahit gaano pa karaming cores meron
        // ang device - isa pa itong sanhi ng sobrang bagal (kasabay ng -O0 debug build).
        // Gamitin natin ang available cores ng device, pero i-cap sa 4 para may
        // matirang core para sa camera/YOLO/face-recognition pipeline na tumatakbo rin
        // nang sabay sa background.
        unsigned int hwThreads = std::thread::hardware_concurrency();
        int n_threads = (int)std::min(std::max(hwThreads, 2u), 4u);
        cparams.n_threads = n_threads;
        cparams.n_threads_batch = n_threads;
        g_ctx = llama_init_from_model(g_model, cparams);
        if (!g_ctx) { jlog(env, "Failed to create context"); return JNI_FALSE; }
        g_n_ctx = (int)cparams.n_ctx;

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
Java_com_example_facerobot_LlamaBridge_generate(JNIEnv* env, jobject, jstring prompt, jstring systemPrompt) {
    // try_lock (hindi blocking lock): kung may isa nang generate() na tumatakbo (dapat
    // hindi na mangyari dahil naka-guard na sa Kotlin side ang llamaBusy), huwag na
    // itong hintayin at magbalik na lang agad ng "" imbis na sabay-sabay silang
    // tumira sa parehong context (dun nangyayari ang crash).
    std::unique_lock<std::mutex> lock(g_mutex, std::try_to_lock);
    if (!lock.owns_lock()) {
        jlog(env, "generate() ignored - may isa pang generate() na tumatakbo");
        return env->NewStringUTF("");
    }
    try {
        if (!g_ctx || !g_model || !g_vocab) {
            jlog(env, "generate() called but model not loaded");
            return env->NewStringUTF("");
        }

        const char* p = env->GetStringUTFChars(prompt, nullptr);
        const char* sysP = env->GetStringUTFChars(systemPrompt, nullptr);
        // Kung walang ipinasang system prompt (blangko), gumamit na lang ng simpleng
        // default para hindi basta mag-crash/mawalan ng context ang generate().
        std::string sysText = (sysP != nullptr && sysP[0] != '\0')
            ? std::string(sysP)
            : "Ikaw ay isang helpful robot assistant. Sumagot nang maikli sa Taglish.";
        std::string fullPrompt =
            "<|start_header_id|>system<|end_header_id|>\n\n" + sysText + "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n" + std::string(p) + "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n";
        env->ReleaseStringUTFChars(prompt, p);
        env->ReleaseStringUTFChars(systemPrompt, sysP);

        // I-clear ang KV cache/memory bago mag-tokenize ng bagong tanong. Kung hindi
        // ito ma-clear, nananatili ang mga token mula sa NAKARAANG generate() call sa
        // context (dahil pareho lang na g_ctx ang ginagamit sa bawat tawag), kaya
        // pinagsasama-sama ang dating system-prompt + tanong + sagot at yung bagong
        // system-prompt + tanong sa iisang context. Yun ang dahilan kung bakit
        // nagre-repeat/nagsu-summarize na lang si Llama ng NAKARAANG tanong imbis na
        // sagutin ang bagong tanong - nalilito siya sa magkakapatong na mga turn.
        llama_memory_t mem = llama_get_memory(g_ctx);
        if (mem) llama_memory_clear(mem, true);

        std::vector<llama_token> tokens = common_tokenize(g_ctx, fullPrompt, true, true);

        // Guard 1: kung walang laman ang tokens (hal. defective tokenizer output o
        // walang laman ang recognized text), i-abort bago pa dumaan sa decode - ang
        // pagpasa ng 0-token batch papunta kay llama_decode ay puwedeng magdulot ng
        // undefined behavior / crash.
        if (tokens.empty()) {
            jlog(env, "Walang nabuong token mula sa prompt, kinansela ang generate()");
            return env->NewStringUTF("");
        }

        // Guard 2: kung mas malaki na ang prompt (system + tanong) kaysa sa context
        // window (n_ctx=512), i-reject muna imbis na hayaang mag-overflow ang KV cache -
        // maraming bersyon ng llama.cpp ang gumagamit ng GGML_ASSERT/abort() sa ganitong
        // sitwasyon, na hindi kayang hulihin ng try/catch dahil hindi ito C++ exception.
        const int maxNewTokens = 80;
        if ((int)tokens.size() + maxNewTokens >= g_n_ctx) {
            jlog(env, "Sobrang haba ng tanong (%d tokens) para sa context window, kinansela",
                 (int)tokens.size());
            return env->NewStringUTF("");
        }

        llama_batch batch = llama_batch_get_one(tokens.data(), (int)tokens.size());
        if (llama_decode(g_ctx, batch) != 0) {
            jlog(env, "Initial decode failed");
            return env->NewStringUTF("");
        }

        std::string result;
        int n_vocab = llama_vocab_n_tokens(g_vocab);
        for (int i = 0; i < maxNewTokens; i++) {
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
