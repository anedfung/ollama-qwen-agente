from agent import preguntar_agente

while True:
  pregunta = input("\nNombre de archivo y pregunta (o 'salir'): ")

  if pregunta.lower() == "salir":
    break

  # Ahora el nombre del archivo se define en la pregunta en vez de ser fijo.
  respuesta = preguntar_agente(pregunta)
  print("\nAgente:", respuesta)