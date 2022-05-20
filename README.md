# bert-api
表格欄位語意辨識的API
1. app.py: 啟動API server的進入點
2. utils: 未來新增model以外的function時使用，例如：logging.py(寫log)
3. pretrained_bert: 裡面放torch模型的相關文件，如：model weight(未上傳), vocabulary等
4. configs: 定義一些常用的參數或常數。
5. base_api: 自定義的class
6. apis/semantic_type: BERT的inferrence code與end point
