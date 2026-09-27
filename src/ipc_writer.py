import mmap
import struct
import time
import random

# tamaño de la memoria (5 floats x 4 bytes)
BUFFER_SIZE = 20
IPC_TAG = "Local\\QuantBuffer"

print(f"Abriendo canal IPC en RAM: {IPC_TAG}...")

# Reservar la memoria compartida en Windows
# El parámetro '0' en Windows le dice que no use un archivo físico, sino el archivo de paginación de la RAM.
shm = mmap.mmap(0, BUFFER_SIZE, tagname=IPC_TAG, access=mmap.ACCESS_WRITE)

try:
    precio_base = 500.0
    while True:
        # Simulamos los 5 datos del tick: price, size, returns, ma_5, ma_15
        precio_base += random.uniform(-0.5, 0.5)
        tick_data = [precio_base, 100.0, 0.001, 499.0, 498.5]
        
        # 3. Serializar (struct.pack)
        # '5f' significa que vamos a empaquetar 5 floats en little-endian
        raw_bytes = struct.pack('5f', *tick_data)
        
        # 4. Inyectar en RAM
        shm.seek(0) # Volver al inicio del bloque de memoria
        shm.write(raw_bytes) # Sobreescribir los 20 bytes
        
        print(f"[TX] Tick inyectado en RAM: {tick_data[0]:.2f}")
        time.sleep(1) # Simulamos 1 tick por segundo de Alpaca
        
except KeyboardInterrupt:
    shm.close()
    print("\nCanal IPC cerrado.")