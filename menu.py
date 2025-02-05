import util

def menu():
    print("Menú de Audio")
    print("1. Grabar audio")
    print("2. Modificar tono y graficar")
    print("3. Agregar eco y graficar")
    print("4. Aplicar distorsión y graficar con filtro")
    print("5. Aplicar chorus y graficar")
    print("6. Salir")

    opcion = input("Ingrese el número de la opción deseada: ")

    if opcion == "1":
        util.grabar_audio("grabacion.wav", 5, 44100)
    elif opcion == "2":
        util.modificar_tono_y_graficar("grabacion.wav", -4)
    elif opcion == "3":
        util.agregar_eco_y_graficar("grabacion.wav", 0.5, 5000)
    elif opcion == "4":
        util.aplicar_distorsion_y_graficar_con_filtro("grabacion.wav", 2, 0.5, 5000, 'lowpass')
    elif opcion == "5":
        util.aplicar_chorus_y_graficar("grabacion.wav", 0.5, 1, 0.005, 0.5)
    elif opcion == "6":
        print("Saliendo...")
    else:
        print("Opción inválida. Por favor, seleccione una opción válida.")

    if opcion != "6":
        menu()

menu()