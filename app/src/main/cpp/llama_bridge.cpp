#include <jni.h>
#include <string>

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_facerobot_LlamaBridge_ping(JNIEnv* env, jobject) {
    return env->NewStringUTF("llama_bridge alive");
}
