## Agente IA Local (Ollama + Qwen)

Este proyecto implementa un agente de IA local usando:

* Modelo local vía Ollama
* Python
* Tool Calling (agente real)
* Herramienta para leer archivos

El agente puede decidir automáticamente cuándo usar herramientas para responder preguntas.

---

### Requisitos

Antes de empezar asegúrate de tener instalado:

* Python 3.10 o superior
* Git
* Ollama → [https://ollama.com](https://ollama.com)

---

### 1. Instalar Ollama y modelo

Instala Ollama y luego descarga un modelo compatible con tools:

```bash
ollama pull qwen2.5:3b
```

Puedes verificar que funciona:

```bash
ollama run qwen2.5:3b
```

---

### 2. Clonar el repositorio

```bash
git clone https://github.com/anedfung/ollama-gemma3-agente.git
cd ollama-gemma3-agente
```

---

### 3. Crear entorno virtual venv (Windows)

#### Instalar venv en directorio de proyecto.

```bash
python -m venv venv
```

#### Comando para activar venv

```bash
venv\Scripts\activate
```

#### Comando para desactivar venv (para después)

```bash
deactivate
```

---

### 4. Instalar dependencias

Si aún no existe, crea el archivo `requirements.txt`:

```txt
ollama
python-dotenv
chromadb
sentence-transformers
uvicorn
rapidapi
pydantic
```

Asegurar que el entorno **venv** esté activado. Si no lo está correr este comando:

```bash
venv/Scripts/activate
```

Luego instala (Puede durar varios minutos):

```bash
python -m pip install -r requirements.txt
```

Verificar que se instalaron las dependencias:
```bash
python -m pip list
```

---

### 5. Ejecutar la aplicación

#### 5.1. Ejecutar en CLI

```bash
python main.py
```

Flujo de indexación y preguntas:

```
ejemplo.txt
Según los documentos en memoria, de que color son las rosas?
```

Salir de la aplicación:

```bash
salir
```

#### 5.2. Ejecutar API

```bash
python -m uvicorn api.app:app --reload
```

Accesar a la UI del API:
```
http://127.0.0.1:8000/docs
```

---