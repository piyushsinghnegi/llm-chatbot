from fastapi import FastAPI, HTTPException
import requests
import os

app = FastAPI()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434/v1/chat/completions")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:0.5b")


@app.post("/chat")
def chat(prompt: str):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        data = response.json()

        if "choices" not in data:
            raise HTTPException(status_code=502, detail="Invalid response from ollama")

        return data

    except HTTPException:
        raise

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="ollama timeout")

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="ollama unavailable")

    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"ollama HTTP error: {str(e)}")

    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
def check_health():
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "ping"}]
            },
            timeout=10
        )

        return {"status": "ok", "ollama_status": response.status_code}

    except requests.exceptions.Timeout:
        return {"status": "error", "reason": "timeout"}

    except requests.exceptions.ConnectionError:
        return {"status": "error", "reason": "unreachable"}

    except Exception:
        return {"status": "error", "reason": "unknown"}
