import ollama
import json
from tools import leer_archivo
from rag import buscar_contexto, indexar_archivo

PLANNER_PROMPT = """
Eres un planificador de agentes IA.

Tu trabajo NO es responder la pregunta.

Tu trabajo es crear un plan paso a paso usando acciones.

Acciones disponibles:

- buscar_memoria(pregunta)
- indexar_archivo(nombre_archivo)
- responder_usuario(texto)

Reglas:

- Si la pregunta requiere información previa,
  primero usa buscar_memoria.
- El último paso SIEMPRE debe ser responder_usuario.
- Devuelve SOLO JSON válido.

Formato:

{
  "plan": [
    {"accion": "...", "argumentos": {...}},
    {"accion": "...", "argumentos": {...}}
  ]
}
"""

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

def crear_plan(pregunta):

  mensajes = [
    {"role": "system", "content": PLANNER_PROMPT},
    {"role": "user", "content": pregunta}
  ]

  respuesta = ollama.chat(
    model="qwen2.5:3b",
    messages=mensajes
  )

  contenido = respuesta["message"]["content"]

  try:
    plan = json.loads(contenido)["plan"]

    # 🔥 GARANTÍA DE RESPUESTA FINAL
    if not any(p["accion"] == "responder_usuario" for p in plan):
      plan.append({
        "accion": "responder_usuario",
        "argumentos": {}
      })

    return plan

  except Exception as e:
    print("Error creando plan:", e)
    print(contenido)
    return []
  
def ejecutar_plan(plan, pregunta):

  contexto = ""

  for paso in plan:

    accion = paso["accion"]
    args = paso.get("argumentos", {})

    print(f"\n⚙️ Ejecutando: {accion}")

    if accion == "buscar_memoria":
      contexto = buscar_contexto(args["pregunta"])

    elif accion == "indexar_archivo":
      ejecutar_tool("indexar_archivo", args)

    elif accion == "responder_usuario":

      mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
          "role": "user",
          "content": f"""
            Pregunta:
            {pregunta}

            Contexto:
            {contexto}
          """
        }
      ]

      respuesta = ollama.chat(
        model="qwen2.5:3b",
        messages=mensajes
      )

      return respuesta["message"]["content"]

  return "No se pudo ejecutar el plan."

def preguntar_agente(pregunta):

  print("\n🧠 Creando plan...")
  plan = crear_plan(pregunta)

  print("\n📋 Plan generado:")
  for paso in plan:
    print(paso)

  respuesta = ejecutar_plan(plan, pregunta)

  return respuesta