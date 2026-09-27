#include <iostream>
#include <windows.h>
#include <onnxruntime_cxx_api.h>

int main() {
    // Conexión a la memoria compartida (IPC) de Windows
    std::cout << "Buscando canal IPC 'Local\\QuantBuffer'..." << std::endl;
    HANDLE hMapFile = OpenFileMappingA(
        FILE_MAP_READ, 
        FALSE, 
        "Local\\QuantBuffer" 
    );

    if (hMapFile == NULL) {
        std::cout << "Error: Script de Python inactivo." << std::endl;
        return 1;
    }

    // Extracción del puntero crudo a la RAM
    float* pBuf = (float*) MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 20);
    if (pBuf == NULL) {
        std::cout << "Error al mapear la memoria." << std::endl;
        CloseHandle(hMapFile);
        return 1;
    }

    // Creamos el entorno dentro de Ort (Namespace de Windows) y definimos que se use solo 1 hilo del procesador
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "QuantEngine");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    
    // En Windows, ONNX exige strings de caracteres anchos (wchar_t) para las rutas
    const wchar_t* model_path = L"../models/quant_model.onnx"; 
    
    std::cout << "Cargando grafo..." << std::endl;
    Ort::Session session(env, model_path, session_options);

    // Estructuras de memoria consecutiva para los datos (1 fila x 5 columnas)
    std::vector<int64_t> input_shape = {1, 5};
    
    // Punteros a memoria. Pasamos el puntero crudo de la RAM (pBuf) al tensor de ONNX (Zero-copy)
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, 
        pBuf, 
        5, 
        input_shape.data(), 
        input_shape.size()
    );

    const char* input_names[] = {"float_input"};
    const char* output_names[] = {"label"}; //'label' es la prediccion final

    std::cout << "Iniciando bucle de vigilancia HFT..." << std::endl;

    float ultimo_precio = 0.0f;
    
    // Bucle de inferencia continuo
    while (true) {
        // Comprobamos si Python ha inyectado un tick nuevo en el puntero base
        if (pBuf[0] != ultimo_precio) {
            ultimo_precio = pBuf[0];
            
            // Inferencia
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr}, 
                input_names, 
                &input_tensor, 
                1, 
                output_names, 
                1
            );

            // Lectura de la salida (Puntero)
            // ONNX devuelve un int64_t para las clasificaciones (0 o 1)
            int64_t* output_data = output_tensors.front().GetTensorMutableData<int64_t>();
            
            std::cout << "Tick: " << pBuf[0] << " | PREDICCION: " << output_data[0] << std::endl;
        }
        // Pausa de 1ms para no saturar la CPU al 100%
        Sleep(1); 
    }

    // Liberamos la memoria compartida y cerramos canales
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);
    return 0;
}