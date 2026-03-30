from fastapi.testclient import TestClient
from unittest.mock import patch, Mock
import requests

from app.main import app

client = TestClient(app)


@patch("app.main.requests.post")
def test_health_success(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response

    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["vllm_status"] == 200


@patch("app.main.requests.post")
def test_health_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "error"
    assert response.json()["reason"] == "timeout"


@patch("app.main.requests.post")
def test_health_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["reason"] == "unreachable"


@patch("app.main.requests.post")
def test_chat_success(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [
            {"message": {"content": "Hello!"}}
        ]
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    response = client.post("/chat", params={"prompt": "Hello"})

    assert response.status_code == 200
    data = response.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["content"] == "Hello!"


@patch("app.main.requests.post")
def test_chat_timeout(mock_post):
    mock_post.side_effect = requests.exceptions.Timeout

    response = client.post("/chat", params={"prompt": "Hello"})

    assert response.status_code == 504
    assert response.json()["detail"] == "vLLM timeout"


@patch("app.main.requests.post")
def test_chat_connection_error(mock_post):
    mock_post.side_effect = requests.exceptions.ConnectionError

    response = client.post("/chat", params={"prompt": "Hello"})

    assert response.status_code == 503
    assert response.json()["detail"] == "vLLM unavailable"


@patch("app.main.requests.post")
def test_chat_http_error(mock_post):
    mock_response = Mock()
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 error")
    mock_post.return_value = mock_response

    response = client.post("/chat", params={"prompt": "Hello"})

    assert response.status_code == 502
    assert "vLLM HTTP error" in response.json()["detail"]


@patch("app.main.requests.post")
def test_chat_invalid_response(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {"unexpected": "data"}
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    response = client.post("/chat", params={"prompt": "Hello"})

    assert response.status_code == 502
    assert response.json()["detail"] == "Invalid response from vLLM"


@patch("app.main.requests.post")
def test_chat_empty_prompt(mock_post):
    mock_response = Mock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": ""}}]
    }
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    response = client.post("/chat", params={"prompt": ""})

    assert response.status_code == 200
