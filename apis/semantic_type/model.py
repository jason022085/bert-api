from flask_restplus import Namespace, Resource, fields, model
from werkzeug.datastructures import FileStorage

api = Namespace("semantic_type", description=u"智慧辨識欄位名稱")

predict_output_payload = api.model('欄位預測參數定義', {
    'result': fields.List(fields.List(fields.String(required = True))),
    'message': fields.String(required=True, default=""),
})

columns_header_input_payload = api.model('欄位表格input payload',{
    'header': fields.List(fields.String(required = True)),
    'table': fields.List(fields.List(fields.String(required = True)))
})