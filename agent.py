import ollama
from tools import leer_archivo

SYSTEM_PROMPT = """
Eres un agente que responde preguntas sobre archivos.
Usa el contenido proporcionado para contestar claramente.
Si no sabes la respuesta, dilo honestamente.
No puedes responder preguntar que no estén relacionadas a los archivos proporcionados.
"""

def preguntar_agente(pregunta: str, archivo: str):

  contenido = leer_archivo(archivo)

  prompt = f"""
    Contenido del archivo:
    {contenido}

    Pregunta del usuario:
    {pregunta}
  """

  respuesta = ollama.chat(
      model="gemma3",
      messages=[
          {"role": "system", "content": SYSTEM_PROMPT},
          {"role": "user", "content": prompt},
      ],
  )

  return respuesta["message"]["content"]