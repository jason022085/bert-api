# -*- coding: UTF-8 -*-
from base_api.custom_cls import (CustomMethodView, CustomRequestParser,
                                 CustomResource)
from apis.semantic_type.api import api as semantic_type_ns #新增api改這行

from flask import Blueprint, Flask, request, session
from flask_jwt_extended import JWTManager
from datetime import datetime, timedelta
from .custom_cls import Api
import os
import logging

api_blueprint = Blueprint('api', __name__, url_prefix='/api')
api = Api(api_blueprint, version="0.0.1", description='', title='Kuohwa API Service', doc="/doc")

# init app
app = Flask(__name__, template_folder="../template", static_folder="../", static_url_path="")

app.config.SWAGGER_UI_DOC_EXPANSION = 'list'
app.config.SWAGGER_UI_REQUEST_DURATION = True
app.secret_key = "test123456789"
app.config['JSON_AS_ASCII'] = False
# app.config['SESSION_CRYPTO_KEY'] = load_aes_key()
# app.config["SESSION_COOKIE_HTTPONLY"] = True
# app.session_interface = EncryptedSessionInterface()


# register blueprint
app.register_blueprint(api_blueprint)

# register swagger api
api.add_namespace(semantic_type_ns) #註冊新api改這行

# setting jwt
JWT_SECRET_KEY = app.config['JWT_SECRET_KEY'] = 'test-secret' 
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(minutes=10)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(minutes=30)


jwt = JWTManager(app)
logging.basicConfig(level=os.getenv('LOG_LEVEL'), format=os.getenv("LOG_FORMAT"))
