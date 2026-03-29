from pathlib import Path

DATA_FOLDER = Path("data")

def leer_archivo(nombre_archivo: str) -> str:
  ruta = DATA_FOLDER / nombre_archivo

  if not ruta.exists():
      return "Archivo no encontrado."

  return ruta.read_text(encoding="utf-8")