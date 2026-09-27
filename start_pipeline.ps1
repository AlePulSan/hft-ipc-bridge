# Script de arranque asíncrono para el entorno HFT
# Levanta la memoria compartida y conecta el motor de inferencia

echo " Iniciando getsor de datos y mapeando RAM..."
Start-Process -FilePath "python" -ArgumentList "src\ipc_writer.py"

# Forzamos un delay de 1s para asegurar que el buffer IPC existe antes de que C++ lo busque
Start-Sleep -Seconds 1

# Comprobación del binario compilado
$engine_path = "build\Debug\quant_engine.exe"

if (Test-Path $engine_path) {
    echo "Conectando punteros y levantando motor C++..."
    Start-Process -FilePath $engine_path
    echo "Pipeline online. Cierra las ventanas de los procesos para detener el sistema."
} else {
    echo "Error: quant_engine.exe no encontrado. Compilar el binario primero."
}