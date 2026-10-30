#ifndef POLICY_NETWORK_H
#define POLICY_NETWORK_H

#include <Arduino.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"

class PolicyNetwork {
private:
    const tflite::Model* model;
    tflite::MicroInterpreter* interpreter;
    TfLiteTensor* input_tensor;
    TfLiteTensor* output_tensor;
    
    // TFLite 연산을 위한 메모리 공간 할당 (모델 크기에 따라 조절 필요)
    static constexpr int kTensorArenaSize = 64 * 1024; // 64KB
    
    // alignas(16)을 추가하면 메모리 접근 속도와 안정성이 향상됩니다.
    alignas(16) uint8_t tensor_arena[kTensorArenaSize];
    //uint8_t* tensor_arena;

    int num_inputs;
    int num_outputs;

public:
    // 1. 생성자 (기본값 초기화)
    PolicyNetwork() 
        : model(nullptr), interpreter(nullptr), 
          input_tensor(nullptr), output_tensor(nullptr),
          //tensor_arena(nullptr),
          num_inputs(0), num_outputs(0) {}

    // 2. 소멸자
    ~PolicyNetwork() {}

    // 3. 모델 초기화 메소드
    bool begin(const unsigned char* model_data) {
        Serial.println("[POLICY] 1. begin START");
        model = tflite::GetModel(model_data);
        Serial.println("[POLICY] 2. GetModel OK");

        Serial.printf("[POLICY] Model version: %d\n", model->version());
        Serial.printf("[POLICY] Expected version: %d\n", TFLITE_SCHEMA_VERSION);

        if (model->version() != TFLITE_SCHEMA_VERSION) {
            Serial.printf("[PolicyNetwork] Schema mismatch! Expected %d, got %d\n",
                          TFLITE_SCHEMA_VERSION, model->version());
            return false;
        }
        Serial.println("[POLICY] 3. Schema OK");
        // TFLite 연산자 등록
        static tflite::AllOpsResolver resolver;
        static tflite::MicroErrorReporter micro_error_reporter;

        //tensor_arena = (uint8_t*)ps_malloc(kTensorArenaSize);
        //if (tensor_arena == nullptr) {
        //    Serial.println("[POLICY] ERROR: PSRAM allocation failed!");
        //    return false;
        //}

        Serial.println("[POLICY] 4. Resolver OK");
        Serial.printf(
            "[POLICY] Tensor arena address: %p\n",
            tensor_arena
        );

        Serial.printf(
            "[POLICY] Tensor arena size: %d bytes\n",
            kTensorArenaSize
        );
        // 인터프리터 생성
        static tflite::MicroInterpreter static_interpreter(
            model, 
            resolver, 
            tensor_arena, 
            kTensorArenaSize,
            &micro_error_reporter,
            nullptr,
            nullptr
        );
        Serial.println("[POLICY] 5. Interpreter created");
        interpreter = &static_interpreter;
        Serial.println("[POLICY] 6. Before AllocateTensors");

        // 텐서 메모리 할당
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            Serial.println("[PolicyNetwork] AllocateTensors() failed! Increase kTensorArenaSize.");
            return false;
        }
        Serial.println("[POLICY] 8. AllocateTensors OK");
        // 입출력 포인터 연결
        input_tensor = interpreter->input(0);
        output_tensor = interpreter->output(0);
        Serial.println("[POLICY] 9. IO tensor OK");
        // 입력/출력 차원 개수 계산 (Float32 기준)
        num_inputs = input_tensor->bytes / sizeof(float);
        num_outputs = output_tensor->bytes / sizeof(float);

        Serial.printf("[PolicyNetwork] Initialized successfully. Inputs: %d, Outputs: %d\n", 
                      num_inputs, num_outputs);
        return true;
    }

    // 4. 추론 실행 메소드
    bool predict(const float* obs, float* action) {
        if (interpreter == nullptr) {
            Serial.println("[PolicyNetwork] Error: Model not initialized! Call begin() first.");
            return false;
        }

        // (1) Observation 데이터를 TFLite 입력 텐서로 복사
        for (int i = 0; i < num_inputs; i++) {
            input_tensor->data.f[i] = obs[i];
        }

        // (2) 인공지능 추론 (Forward Pass) 실행
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("[PolicyNetwork] Inference Invoke failed!");
            return false;
        }

        // (3) TFLite 출력 텐서에서 결과를 Action 배열로 복사
        for (int i = 0; i < num_outputs; i++) {
            action[i] = output_tensor->data.f[i];
        }

        return true;
    }

    // 5. Getters
    int getNumInputs() const { return num_inputs; }
    int getNumOutputs() const { return num_outputs; }
};

#endif // POLICY_NETWORK_H