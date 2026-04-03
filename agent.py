import ollama
import json
from tools import leer_archivo
from rag import buscar_contexto, indexar_archivo

from dataclasses import dataclass

@dataclass
class AgentState:
    pregunta: str
    plan: list | None = None
    respuesta: str | None = None
    contexto: str | None = None
    correcto: bool = False
    intentos: int = 0

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

EVALUATOR_PROMPT = """
Eres un evaluador de respuestas de un agente IA.

Debes analizar si la respuesta es correcta
según la pregunta original y el contexto recuperado.

Reglas:

- Si la pregunta requiere información previa,
  la respuesta DEBE usar el contexto.
- Si no hay contexto cuando debería haberlo,
  la respuesta es incorrecta.
- Sé estricto.

Devuelve SOLO JSON válido:

{
  "correcto": true
}

o

{
  "correcto": false,
  "razon": "explicación breve"
}
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


## Funciones de tools

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

      return respuesta["message"]["content"], contexto

  return "No se pudo ejecutar el plan."

def evaluar_respuesta(pregunta, respuesta, contexto):

  mensajes = [
    {"role": "system", "content": EVALUATOR_PROMPT},
    {
      "role": "user",
      "content": f"""
        Pregunta:
        {pregunta}

        Contexto:
        {contexto}

        Respuesta del agente:
        {respuesta}
        """
    }
  ]

  evaluacion = ollama.chat(
    model="qwen2.5:3b",
    messages=mensajes
  )

  contenido = evaluacion["message"]["content"]

  try:
    return json.loads(contenido)
  except:
    print("Error evaluando:")
    print(contenido)
    return {"correcto": True}
  

## Funciones de State Machine

def nodo_plan(state: AgentState):

  print("\n🧠 [PLAN]")

  state.plan = crear_plan(state.pregunta)

  for paso in state.plan:
    print(paso)

  return "execute", state

def nodo_execute(state: AgentState):

  print("\n⚙️ [EXECUTE]")

  respuesta, contexto = ejecutar_plan(
    state.plan,
    state.pregunta
  )

  state.respuesta = respuesta
  state.contexto = contexto

  return "evaluate", state
  
def nodo_evaluate(state: AgentState):

  print("\n🧪 [EVALUATE]")

  evaluacion = evaluar_respuesta(
    state.pregunta,
    state.respuesta,
    state.contexto
  )

  state.correcto = evaluacion.get("correcto", True)
  state.intentos += 1

  if state.correcto:
    return "end", state

  if state.intentos >= 2:
    print("⚠️ Máximo de intentos alcanzado")
    return "end", state

  print("🔁 Replaneando...")
  return "plan", state

def ejecutar_grafo(state: AgentState):

  nodos = {
    "plan": nodo_plan,
    "execute": nodo_execute,
    "evaluate": nodo_evaluate,
  }

  nodo_actual = "plan"

  while nodo_actual != "end":
    nodo_actual, state = nodos[nodo_actual](state)

  return state

def preguntar_agente(pregunta):

    estado_inicial = AgentState(pregunta=pregunta)

    estado_final = ejecutar_grafo(estado_inicial)

    return estado_final.respuesta