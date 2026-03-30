import ollama
import json
from tools import leer_archivo

SYSTEM_PROMPT = """
Eres un agente que responde preguntas sobre archivos.
Usa el contenido proporcionado para contestar claramente.
Si no sabes la respuesta, dilo honestamente.
No puedes responder preguntar que no estén relacionadas a los archivos proporcionados.
"""

TOOLS = [
  {
    "type": "function",
    "function": {
      "name": "leer_archivo",
      "description": "Lee el contenido de un archivo",
      "parameters": {
        "type": "object",
        "properties": {
          "nombre_archivo": {"type": "string"},
        },
        "required": ["nombre_archivo"],
      },
    },
  }
]

def ejecutar_tool(nombre, argumentos):
  if nombre == "leer_archivo":
    return leer_archivo(argumentos["nombre_archivo"])

  return "Tool desconocida"

def preguntar_agente(pregunta):

  # Se separa messages para reutilizar.
  messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": pregunta},
  ]

  while True:

    respuesta = ollama.chat(
      model="qwen2.5:3b",
      messages=messages,
      tools=TOOLS,
    )

    # Separa respuesta["message"] para no usar las demas propiedades de respuesta.
    message = respuesta["message"]
    messages.append(message)

    # Permite ver el procesamiento del agente en cada repeticion.
    print("LLM:", message)

    # Primero no contiene tool_calls, entonces busca el tool indicado y obtiene el resultado.
    if "tool_calls" in message:

      for tool_call in message["tool_calls"]:
        nombre = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        resultado = ejecutar_tool(nombre, args)

        # Se agrega el resultado a messages
        messages.append({
          "role": "tool",
          "content": resultado,
        })

    # Segundo ya tiene tool_calls, entonces devuelve el resultado.
    else:
      return message["content"]