from fastapi import FastAPI, HTTPException
import requests
import os

app = FastAPI()

VLLM_URL = os.getenv("VLLM_URL", "http://vllm:8000/v1/chat/completions")


@app.post("/chat")
def chat(prompt: str):
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            },
            timeout=50
        )

        response.raise_for_status()

        data = response.json()

        if "choices" not in data:
            raise HTTPException(status_code=502, detail="Invalid response from vLLM")

        return data

    except HTTPException:
        raise

    except requests.exceptions.Timeout:
        raise HTTPException(status_code=504, detail="vLLM timeout")

    except requests.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="vLLM unavailable")

    except requests.exceptions.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"vLLM HTTP error: {str(e)}")

    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/health")
def check_health():
    try:
        response = requests.post(
            VLLM_URL,
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "messages": [{"role": "user", "content": "ping"}]
            },
            timeout=5
        )

        return {"status": "ok", "vllm_status": response.status_code}

    except requests.exceptions.Timeout:
        return {"status": "error", "reason": "timeout"}

    except requests.exceptions.ConnectionError:
        return {"status": "error", "reason": "unreachable"}

    except Exception:
        return {"status": "error", "reason": "unknown"}