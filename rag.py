import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# modelo de embeddings (local)
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# base persistente (se guarda en disco)
client = chromadb.Client(
	chromadb.config.Settings(
		persist_directory="memory",
		anonymized_telemetry=False
	)
)

collection = client.get_or_create_collection("documentos")

def embed(texto: str):
	return embedding_model.encode(texto).tolist()

def indexar_archivo(nombre_archivo: str):

	ruta = Path("data") / nombre_archivo

	if not ruta.exists():
		return "Archivo no encontrado"

	texto = ruta.read_text(encoding="utf-8")

	chunks = dividir_texto(texto)

	for i, chunk in enumerate(chunks):
		collection.add(
			ids=[f"{nombre_archivo}_{i}"],
			documents=[chunk],
			embeddings=[embed(chunk)]
		)

	return f"{len(chunks)} partes indexadas."


def dividir_texto(texto, tamaño=500):
	return [
		texto[i:i+tamaño]
		for i in range(0, len(texto), tamaño)
	]

def buscar_contexto(pregunta: str, k=3):

	resultados = collection.query(
		query_embeddings=[embed(pregunta)],
		n_results=k
	)

	docs = resultados["documents"][0]

	return "\n\n".join(docs)