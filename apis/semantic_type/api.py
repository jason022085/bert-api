from email import header
from apis.semantic_type.model import api, predict_output_payload, columns_header_input_payload
from apis.semantic_type.module import Prediction
from flask import session, abort, request
from base_api import CustomResource
from pathlib import Path
from flask_restplus.reqparse import HTTPStatus
from flask_restplus.errors import abort
import logging
import pandas as pd

@api.route("/predict_semantic_type")
class SemanticType(CustomResource):
    @api.expect(columns_header_input_payload)
    @api.marshal_with(predict_output_payload)
    def post(self):
        """使用BERT進行辨識"""
        data = api.payload
        dataframe = pd.DataFrame(data["table"], columns = data["header"])
        return Prediction.evaluate(dataframe)
