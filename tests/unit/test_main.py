from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "Hello World"}


def test_say_hello() -> None:
    response = client.get("/hello/Ada")

    assert response.status_code == 200
    assert response.json() == {"message": "Hello Ada"}
