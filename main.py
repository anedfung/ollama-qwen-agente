from agent import preguntar_agente

while True:
  pregunta = input("\nPregunta (o 'salir'): ")

  if pregunta.lower() == "salir":
    break

  respuesta = preguntar_agente(pregunta, "ejemplo.txt")
  print("\nAgente:", respuesta)