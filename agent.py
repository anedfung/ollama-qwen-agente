import ollama
import json
from tools import leer_archivo
from rag import buscar_contexto, indexar_archivo

SYSTEM_PROMPT = """
Eres un agente con memoria persistente.

Tienes estas capacidades:

1. indexar_archivo(nombre_archivo)
   - Guarda un documento en tu memoria permanente.

2. buscar_memoria(pregunta)
   - Busca información relevante en documentos previamente indexados.

REGLAS IMPORTANTES:

- Cuando el usuario haga preguntas sobre documentos,
  SIEMPRE debes usar primero buscar_memoria.
- No pidas el nombre del archivo si puedes encontrar
  la información usando buscar_memoria.
- Usa buscar_memoria incluso si el usuario no menciona
  explícitamente un archivo.
- Después de obtener resultados de memoria,
  usa esa información para responder.

Nunca inventes información si puedes buscar en memoria primero.
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
  },
  {
    "type": "function",
    "function": {
      "name": "indexar_archivo",
      "description": "Guarda un archivo en la memoria del agente",
      "parameters": {
        "type": "object",
        "properties": {
          "nombre_archivo": {"type": "string"}
        },
        "required": ["nombre_archivo"]
      }
    }
  },
  {
    "type": "function",
    "function": {
      "name": "buscar_memoria",
      "description": (
        "Busca información relevante en TODOS los documentos "
        "indexados previamente. Debe usarse cuando el usuario "
        "haga preguntas sobre contenido, documentos o información pasada."
      ),
      "parameters": {
        "type": "object",
        "properties": {
            "pregunta": {"type": "string"}
        },
        "required": ["pregunta"]
      }
    }
  }
]

def ejecutar_tool(nombre, argumentos):
  if nombre == "leer_archivo":
    return leer_archivo(argumentos["nombre_archivo"])
  elif nombre == "indexar_archivo":
    return indexar_archivo(argumentos["nombre_archivo"])
  elif nombre == "buscar_memoria":
    return buscar_contexto(argumentos["pregunta"])

  return "Tool desconocida"

def preguntar_agente(pregunta):

  contexto = buscar_contexto(pregunta)

  messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
      "role": "user",
      "content": f"""
        Pregunta del usuario:
        {pregunta}

        Contexto relevante encontrado:
        {contexto}
        """
    },
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
    # print("LLM:", message)

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