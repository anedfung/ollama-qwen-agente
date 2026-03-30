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

#### Comando para desactivar venv

```bash
deactivate
```

---

### 4. Instalar dependencias

Si aún no existe, crea el archivo `requirements.txt`:

```txt
ollama
python-dotenv
```

Luego instala:

```bash
pip install -r requirements.txt
```

---

### 5. Ejecutar la aplicación

```bash
python main.py
```

Ejemplos de preguntas:

```
Que dice el archivo ejemplo.txt?
Como responderias al contenido del archivo ejemplo.txt?
Crees que el contenido del archivo ejemplo.txt es formal o informal?
```

Salir de la aplicación:

```
salir
```

---