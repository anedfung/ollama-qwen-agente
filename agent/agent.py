import ollama
import json
from agent.tools import leer_archivo
from agent.rag import buscar_contexto, indexar_archivo

from dataclasses import dataclass


## Agent State

@dataclass
class AgentState:
    pregunta: str
    plan: list | None = None
    respuesta: str | None = None
    contexto: str | None = None
    correcto: bool = False
    intentos: int = 0

AGENT_STATE = {
    "ultimo_contexto": "",
    "ultima_pregunta": ""
}


## PROMPTS de nodos

CONTEXT_EVALUATOR_PROMPT = """
  Eres un evaluador de relevancia para un sistema RAG.

  IMPORTANTE:
  - NO uses conocimiento del mundo.
  - NO juzgues si la información es verdadera o falsa.
  - SOLO evalúa si el contexto contiene información
    que permita responder la pregunta.

  Si el contexto menciona una posible respuesta,
  aunque sea incorrecta en la vida real,
  debe considerarse relevante.

  Responde SOLO en JSON:

  {
    "relevante": true o false,
    "razon": "explicación corta"
  }
"""

ROUTER_PROMPT = """
  Eres un router de un agente IA.

  Clasifica la intención del usuario.

  Opciones posibles:

  - "chat" → conversación normal sin memoria
  - "memory" → preguntas sobre documentos o información previa
  - "index" → el usuario quiere guardar/indexar un archivo

  Devuelve SOLO JSON válido:

  {
    "ruta": "chat"
  }

  o

  {
    "ruta": "memory"
  }

  o

  {
    "ruta": "index"
  }
"""

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

  IMPORTANTE:
  - NO reformules la pregunta del usuario.
  - Usa exactamente la pregunta recibida.

  Formato:

  {
    "plan": [
      {"accion": "...", "argumentos": {...}},
      {"accion": "...", "argumentos": {...}}
    ]
  }
"""

SYSTEM_PROMPT = """
Eres un agente que responde EXCLUSIVAMENTE usando el contexto.

- Nunca contradigas el contexto.
- Trata el contexto como si fuera un documento legal obligatorio.
- NO puedes usar informacion de la vida real.
- SOLO devuelve la respuesta, sin mencionar que es así segun el contexto.
- Devuelve la respuesta completa.
- Si no hay información suficiente, dilo claramente.

Debes ser conciso y preciso.
"""

RESPONSE_EVALUATOR_PROMPT = """
Eres un evaluador de grounding en un sistema RAG.

Debes decidir si la respuesta está apoyada por el contexto.

REGLAS:

- El contexto es la única fuente de verdad.
- La respuesta puede reformular, resumir o parafrasear el contexto.
- NO requiere coincidencia literal.
- Si la información principal aparece en el contexto,
  la respuesta es correcta.

Ignora conocimiento del mundo real.

Responde SOLO JSON:

{"correcto": true}

o

{"correcto": false, "razon": "..."}
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


## Funciones de nodos

def evaluar_contexto(pregunta, contexto):

  mensajes = [
    {
      "role": "system",
      "content": CONTEXT_EVALUATOR_PROMPT,
    },
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

  evaluacion = ollama.chat(
    model="qwen2.5:3b",
    messages=mensajes
  )

  contenido = evaluacion["message"]["content"]

  print("evaluacion de contexto", contenido)

  try:
    return json.loads(contenido)
  except:
    print("Error evaluando contexto:")
    print(contenido)
    return {"relevante": True}

def hay_memoria_relevante(pregunta):

  contexto = buscar_contexto(pregunta)

  if not contexto.strip():
    return False

  eval_ctx = evaluar_contexto(pregunta, contexto)

  return eval_ctx.get("relevante", False)

def decidir_ruta(pregunta):

  mensajes = [
    {"role": "system", "content": ROUTER_PROMPT},
    {"role": "user", "content": pregunta},
  ]

  respuesta = ollama.chat(
    model="qwen2.5:3b",
    messages=mensajes
  )

  contenido = respuesta["message"]["content"]

  try:
    decision = json.loads(contenido)
    return decision.get("ruta", "chat")
  except:
    print("Router inválido:", contenido)
    return "chat"

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
      # 🔒 usar SIEMPRE la pregunta real del usuario
      query_memoria = pregunta
      if pregunta_relacionada(pregunta) and AGENT_STATE["ultimo_contexto"]:
          print("🧷 Usando sticky context")
          contexto = AGENT_STATE["ultimo_contexto"]
      else:
          contexto = buscar_contexto(query_memoria)

    elif accion == "indexar_archivo":
      ejecutar_tool("indexar_archivo", args)

    elif accion == "responder_usuario":

      historial = ""

      if AGENT_STATE["ultima_pregunta"]:
        historial = f"""
      Pregunta anterior:
      {AGENT_STATE["ultima_pregunta"]}
      """

      mensajes = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
          "role": "user",
          "content": f"""
          {historial}

          Pregunta actual:
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

      respuesta_texto = respuesta["message"]["content"]

      return respuesta_texto, contexto

  return "No se pudo ejecutar el plan."

def evaluar_respuesta(pregunta, respuesta, contexto):

  mensajes = [
    {"role": "system", "content": RESPONSE_EVALUATOR_PROMPT},
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

  print("evaluacion de respuesta:", contenido)

  # if contenido["correcto"]:
  #   AGENT_STATE["ultimo_contexto"] = contexto
  #   AGENT_STATE["ultima_pregunta"] = pregunta

  try:
    return json.loads(contenido)
  except:
    print("Error evaluando:")
    print(contenido)
    return {"correcto": True}
  
def pregunta_relacionada(pregunta_nueva):
  ultima = AGENT_STATE["ultima_pregunta"]

  if not ultima:
    return False

  mensajes = [
    {
      "role": "system",
      "content": """
        Decide si la nueva pregunta continúa el mismo tema
        que la pregunta anterior.

        Responde SOLO:
        SI
        o
        NO
      """
    },
    {
      "role": "user",
      "content": f"""
        Pregunta anterior:
        {ultima}

        Nueva pregunta:
        {pregunta_nueva}
      """
    }
  ]

  r = ollama.chat(
    model="qwen2.5:3b",
    messages=mensajes
  )

  return "SI" in r["message"]["content"].upper()
  

## Funciones de State Machine

def nodo_router(state: AgentState):

  print("\n🧭 [ROUTER]")

  pregunta = state.pregunta.strip()

  # 1️⃣ INDEX
  if pregunta.lower().endswith(".txt"):
    print("Ruta elegida: index")
    return "index", state

  # 2️⃣ ⭐ CONTINUIDAD CONVERSACIONAL (PRIORIDAD MÁXIMA)
  if pregunta_relacionada(pregunta):
    print("🧷 Continuación detectada → memory")
    return "plan", state

  # 3️⃣ memoria semántica
  if hay_memoria_relevante(pregunta):
    print("Ruta elegida: memory (auto)")
    return "plan", state

  # 4️⃣ router LLM
  ruta = decidir_ruta(pregunta)

  print("Ruta elegida:", ruta)

  if ruta == "chat":
    return "chat", state

  if ruta == "index":
    return "index", state

  return "plan", state

def nodo_chat(state: AgentState):

  print("\n💬 [CHAT DIRECTO]")

  respuesta = ollama.chat(
    model="qwen2.5:3b",
    messages=[
      {"role": "user", "content": state.pregunta}
    ]
  )

  state.respuesta = respuesta["message"]["content"]

  return "end", state

def nodo_index(state: AgentState):

  print("\n📚 [INDEXAR]")
  print(state.pregunta.strip());
  resultado = indexar_archivo(state.pregunta.strip())

  state.respuesta = resultado

  return "end", state

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

  # ⭐ GUARDAR CONTEXTO SI FUNCIONÓ
  if state.correcto and state.contexto:
    AGENT_STATE["ultimo_contexto"] = state.contexto
    AGENT_STATE["ultima_pregunta"] = state.pregunta
    print("🧷 Sticky context actualizado")

  if state.correcto:
    return "end", state

  if state.intentos >= 2:
    print("⚠️ Máximo de intentos alcanzado")
    return "end", state

  print("🔁 Replaneando...")
  return "plan", state

def ejecutar_grafo(state: AgentState):

  nodos = {
    "router": nodo_router,
    "chat": nodo_chat,
    "index": nodo_index,
    "plan": nodo_plan,
    "execute": nodo_execute,
    "evaluate": nodo_evaluate,
  }

  nodo_actual = "router"

  while nodo_actual != "end":
    nodo_actual, state = nodos[nodo_actual](state)

  return state

def preguntar_agente(pregunta):

  estado = AgentState(pregunta=pregunta)

  final = ejecutar_grafo(estado)

  return final.respuesta