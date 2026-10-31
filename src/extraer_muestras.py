"""Extrae una muestra de imágenes del dataset original de Microsoft
(Kaggle Cats and Dogs / Asirra) para la galería de la interfaz web.

Uso:  python src/extraer_muestras.py [ruta_al_zip] [cantidad_por_clase]
Por defecto busca el ZIP que descargó tensorflow_datasets.
"""
import glob
import json
import os
import random
import sys
import zipfile

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESTINO = os.path.join(RAIZ, "web", "muestras")
ZIP_POR_DEFECTO = os.path.join(
    os.path.expanduser("~"), "tensorflow_datasets", "downloads", "cats_vs_dogs", "*.zip")


def main():
    ruta_zip = sys.argv[1] if len(sys.argv) > 1 else (glob.glob(ZIP_POR_DEFECTO) or [None])[0]
    cantidad = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    if not ruta_zip or not os.path.exists(ruta_zip):
        sys.exit("No se encontró el ZIP de Microsoft (kagglecatsanddogs_5340.zip).")

    os.makedirs(DESTINO, exist_ok=True)
    random.seed(42)
    lista = []
    with zipfile.ZipFile(ruta_zip) as z:
        for carpeta, etiqueta in (("Cat", "gato"), ("Dog", "perro")):
            nombres = [n for n in z.namelist()
                       if n.startswith(f"PetImages/{carpeta}/") and n.lower().endswith(".jpg")]
            elegidas = 0
            for nombre in random.sample(nombres, len(nombres)):
                datos = z.read(nombre)
                # Algunas imágenes del dataset están corruptas o no son JFIF
                if len(datos) < 8000 or b"JFIF" not in datos[:10]:
                    continue
                archivo = f"{etiqueta}_{os.path.basename(nombre)}"
                with open(os.path.join(DESTINO, archivo), "wb") as f:
                    f.write(datos)
                lista.append({"archivo": archivo, "etiqueta": etiqueta, "original": nombre})
                elegidas += 1
                if elegidas == cantidad:
                    break

    random.shuffle(lista)
    with open(os.path.join(DESTINO, "muestras.json"), "w", encoding="utf-8") as f:
        json.dump({
            "fuente": "Microsoft — Kaggle Cats and Dogs Dataset (Asirra)",
            "url": "https://www.microsoft.com/en-us/download/details.aspx?id=54765",
            "imagenes": lista,
        }, f, ensure_ascii=False, indent=2)
    print(f"{len(lista)} imágenes copiadas a {DESTINO}")


if __name__ == "__main__":
    main()
