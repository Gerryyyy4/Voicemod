import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import sounddevice as sd
from scipy import signal

def grabar_audio(ruta_archivo, duracion, frecuencia_muestreo):
    # Grabar audio
    print("Grabando audio...")
    grabacion = sd.rec(int(duracion * frecuencia_muestreo), samplerate=frecuencia_muestreo, channels=1, dtype='float32')
    sd.wait()

    # Guardar audio en archivo
    sf.write(ruta_archivo, grabacion, frecuencia_muestreo)
    print(f"Grabación guardada en: {ruta_archivo}")

    return ruta_archivo

def modificar_tono_y_graficar(ruta_archivo, semitonos):
    # Cargar la señal de audio
    datos, frecuencia_muestreo = librosa.load(ruta_archivo, sr=None)
    
    # Modificar el tono de la señal
    datos_modificados = librosa.effects.pitch_shift(datos, sr=frecuencia_muestreo, n_steps=semitonos)
    
    # Crear el eje de tiempo para ambas señales
    tiempo = np.linspace(0, len(datos) / frecuencia_muestreo, num=len(datos))
    tiempo_modificado = np.linspace(0, len(datos_modificados) / frecuencia_muestreo, num=len(datos_modificados))
    
    # Crear la figura y los subplots
    plt.figure(figsize=(12, 10))
    
    # Subplot para la señal original en el dominio del tiempo
    plt.subplot(4, 2, 1)
    plt.plot(tiempo, datos)
    plt.title('Señal de Audio Original (Tiempo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la FFT de la señal original
    plt.subplot(4, 2, 2)
    fft_original = np.fft.fft(datos)
    fft_frecuencias = np.fft.fftfreq(len(datos), d=1/frecuencia_muestreo)
    plt.plot(fft_frecuencias, np.abs(fft_original))
    plt.title('FFT de la Señal Original (Frecuencia)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la señal modificada en el dominio del tiempo
    plt.subplot(4, 2, 3)
    plt.plot(tiempo_modificado, datos_modificados)
    plt.title('Señal de Audio Modificada (Tono) (Tiempo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la FFT de la señal modificada
    plt.subplot(4, 2, 4)
    fft_modificada = np.fft.fft(datos_modificados)
    plt.plot(fft_frecuencias, np.abs(fft_modificada))
    plt.title('FFT de la Señal Modificada (Tono) (Frecuencia)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Guardar la señal modificada en un nuevo archivo
    archivo_modificado = 'modificado_' + ruta_archivo
    sf.write(archivo_modificado, datos_modificados, frecuencia_muestreo)
    
    print(f'Tono modificado guardado en: {archivo_modificado}')
    
    return archivo_modificado

def agregar_eco_y_graficar(ruta_archivo, factor_reverberacion, retardo):
    # Cargar la señal de audio
    datos, frecuencia_muestreo = librosa.load(ruta_archivo, sr=None)
    
    # Calcular la señal de eco
    eco = np.zeros_like(datos)
    eco[retardo:] = datos[:-retardo] * factor_reverberacion
    
    # Sumar la señal original con el eco
    datos_con_eco = datos + eco
    
    # Crear el eje de tiempo para ambas señales
    tiempo = np.linspace(0, len(datos) / frecuencia_muestreo, num=len(datos))
    tiempo_con_eco = np.linspace(0, len(datos_con_eco) / frecuencia_muestreo, num=len(datos_con_eco))
    
    # Crear la figura y los subplots
    plt.figure(figsize=(12, 10))
    
    # Subplot para la señal original en el dominio del tiempo
    plt.subplot(4, 2, 1)
    plt.plot(tiempo, datos)
    plt.title('Señal de Audio Original (Tiempo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la FFT de la señal original
    plt.subplot(4, 2, 2)
    fft_original = np.fft.fft(datos)
    fft_frecuencias = np.fft.fftfreq(len(datos), d=1/frecuencia_muestreo)
    plt.plot(fft_frecuencias, np.abs(fft_original))
    plt.title('FFT de la Señal Original (Frecuencia)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la señal con eco en el dominio del tiempo
    plt.subplot(4, 2, 3)
    plt.plot(tiempo_con_eco, datos_con_eco)
    plt.title('Señal de Audio con Eco (Tiempo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la FFT de la señal con eco
    plt.subplot(4, 2, 4)
    fft_con_eco = np.fft.fft(datos_con_eco)
    plt.plot(fft_frecuencias, np.abs(fft_con_eco))
    plt.title('FFT de la Señal con Eco (Frecuencia)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Guardar la señal con eco en un nuevo archivo
    archivo_con_eco = 'con_eco_' + ruta_archivo
    sf.write(archivo_con_eco, datos_con_eco, frecuencia_muestreo)
    
    print(f'Señal con eco guardada en: {archivo_con_eco}')
    
    return archivo_con_eco

def aplicar_distorsion_y_graficar_con_filtro(ruta_archivo, ganancia, umbral, frecuencia_corte, tipo_filtro='lowpass'):
    # Cargar la señal de audio
    datos, frecuencia_muestreo = librosa.load(ruta_archivo, sr=None)
    
    # Aplicar el filtro de pasa bajos o pasa altos
    if tipo_filtro == 'lowpass':
        b, a = signal.butter(4, frecuencia_corte / (frecuencia_muestreo / 2), 'low')
    elif tipo_filtro == 'highpass':
        b, a = signal.butter(4, frecuencia_corte / (frecuencia_muestreo / 2), 'high')
    datos_filtrados = signal.filtfilt(b, a, datos)
    
    # Aplicar la distorsión
    datos_distorsionados = np.tanh(ganancia * datos_filtrados / np.max(np.abs(datos_filtrados)))
    
    # Aplicar umbral si es necesario
    if umbral is not None:
        datos_distorsionados[datos_distorsionados > umbral] = umbral
        datos_distorsionados[datos_distorsionados < -umbral] = -umbral
    
    # Crear el eje de tiempo para ambas señales
    tiempo = np.linspace(0, len(datos) / frecuencia_muestreo, num=len(datos))
    tiempo_distorsionado = np.linspace(0, len(datos_distorsionados) / frecuencia_muestreo, num=len(datos_distorsionados))
    
    # Crear la figura y los subplots
    plt.figure(figsize=(12, 10))
    
    # Subplot para la señal original en el dominio del tiempo
    plt.subplot(4, 2, 1)
    plt.plot(tiempo, datos)
    plt.title('Señal de Audio Original (Tiempo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la FFT de la señal original
    plt.subplot(4, 2, 2)
    fft_original = np.fft.fft(datos)
    fft_frecuencias = np.fft.fftfreq(len(datos), d=1/frecuencia_muestreo)
    plt.plot(fft_frecuencias, np.abs(fft_original))
    plt.title('FFT de la Señal Original (Frecuencia)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la señal distorsionada en el dominio del tiempo
    plt.subplot(4, 2, 3)
    plt.plot(tiempo_distorsionado, datos_distorsionados)
    plt.title('Señal de Audio Distorsionada (Tiempo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    # Subplot para la FFT de la señal distorsionada
    plt.subplot(4, 2, 4)
    fft_distorsionada = np.fft.fft(datos_distorsionados)
    plt.plot(fft_frecuencias, np.abs(fft_distorsionada))
    plt.title('FFT de la Señal Distorsionada (Frecuencia)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Guardar la señal distorsionada en un nuevo archivo
    archivo_distorsionado = 'distorsionado_filtrado_' + ruta_archivo
    sf.write(archivo_distorsionado, datos_distorsionados, frecuencia_muestreo)
    
    print(f'Señal distorsionada guardada en: {archivo_distorsionado}')
    
    return archivo_distorsionado

def aplicar_chorus_y_graficar(ruta_archivo, profundidad, frecuencia, duracion, mezcla):
    # Cargar la señal de audio
    datos, frecuencia_muestreo = librosa.load(ruta_archivo, sr=None)
    
    # Parámetros del efecto de chorus
    cantidad_muestras = len(datos)
    tiempo = np.arange(cantidad_muestras) / frecuencia_muestreo
    modulador = np.sin(2 * np.pi * frecuencia * tiempo)
    
    # Aplicar el efecto de chorus
    efecto_chorus = np.zeros_like(datos)
    for i in range(cantidad_muestras):
        delay = int(duracion * frecuencia_muestreo * (1 + profundidad * modulador[i]))
        efecto_chorus[i] = datos[i] + mezcla * datos[(i - delay) % cantidad_muestras]
    
    # Normalizar la señal resultante
    efecto_chorus /= np.max(np.abs(efecto_chorus))
    
    # Crear el eje de tiempo para ambas señales
    tiempo = np.linspace(0, len(datos) / frecuencia_muestreo, num=len(datos))
    tiempo_chorus = np.linspace(0, len(efecto_chorus) / frecuencia_muestreo, num=len(efecto_chorus))
    
    # Crear la figura y los subplots
    plt.figure(figsize=(12, 10))
    
    # Subplot para la señal original y el efecto de chorus en el dominio del tiempo
    plt.subplot(2, 2, 1)
    plt.plot(tiempo, datos, label='Original')
    plt.title('Señal de Audio Original (Tiempo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(tiempo_chorus, efecto_chorus, color='orange', label='Chorus')
    plt.title('Señal con Efecto de Chorus (Tiempo)')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    # Calcular la transformada de Fourier de ambas señales
    fft_original = np.fft.fft(datos)
    fft_frecuencias = np.fft.fftfreq(len(datos), d=1/frecuencia_muestreo)
    fft_chorus = np.fft.fft(efecto_chorus)
    
    # Subplot para la FFT de la señal original y el efecto de chorus
    plt.subplot(2, 2, 3)
    plt.plot(fft_frecuencias, np.abs(fft_original), label='Original')
    plt.title('FFT de la Señal Original (Frecuencia)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(fft_frecuencias, np.abs(fft_chorus), color='orange', label='Chorus')
    plt.title('FFT de la Señal con Efecto de Chorus (Frecuencia)')
    plt.xlabel('Frecuencia [Hz]')
    plt.ylabel('Amplitud')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Guardar la señal con efecto de chorus en un nuevo archivo
    archivo_chorus = 'chorus_' + ruta_archivo
    sf.write(archivo_chorus, efecto_chorus, frecuencia_muestreo)
    
    print(f'Señal con efecto de chorus guardada en: {archivo_chorus}')
    
    return archivo_chorus

