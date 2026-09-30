# hft-ipc-bridge

Pipeline de inferencia HFT (High Frequency Trading) que aísla la carga analítica de la ejecución a bajo nivel. Desacopla Python y C++ mediante memoria compartida nativa (`mmap`), eliminando la latencia de red y de memoria.

## Arquitectura

- **Data Plane (C++ / ONNX):** Binario compilado y bloqueado a un único hilo de CPU (`SetIntraOpNumThreads(1)`). Lee los tensores inyectando punteros directamente a la RAM (`MapViewOfFile`) y evalúa el grafo matemático en crudo.
- **Control Plane (Python):** Script asíncrono que ingesta el mercado, vectoriza las features (Pandas) y reescribe el buffer IPC con cada nuevo *tick*.

```mermaid
graph LR
    subgraph Control_Plane [Python - Lógica]
        A[Ingesta de Mercado] --> B[Feature Engineering]
        B --> C[IPC Writer]
        ML[GitHub Actions] -.->|Actualiza| ONNX[(quant_model.onnx)]
    end

    subgraph IPC [Puente SO]
        C == Escribe ==> RAM{Shared RAM <br> Local\QuantBuffer}
    end

    subgraph Data_Plane [C++ - Ejecución]
        RAM == Puntero Físico ==> Engine[Motor ONNX]
        ONNX -.->|Grafo Estático| Engine
        Engine --> OUT[Señal: 0 / 1]
    end
    
    style Control_Plane fill:#2b323b,stroke:#4a5568,stroke-width:2px,color:#fff
    style Data_Plane fill:#1e3a5f,stroke:#2b6cb0,stroke-width:2px,color:#fff
    style IPC fill:#1a202c,stroke:#718096,stroke-width:2px,color:#fff
    style RAM fill:#38a169,stroke:#22543d,color:#fff
    style ONNX fill:#d69e2e,stroke:#975a16,color:#fff
