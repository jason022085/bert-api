import os


def of(from_env, default):
    value = os.getenv(from_env)
    return value if value else default
    
FLASK_IP = of(from_env="FLASK_IP", default="127.0.0.1")
FLASK_PORT = of(from_env="FLASK_PORT", default=5000)