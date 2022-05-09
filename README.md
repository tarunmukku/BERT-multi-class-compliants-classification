**
Demo of the application : **
https://www.youtube.com/watch?v=VIkHNG0aM50


Transformer based deep learning classification model for classifying costumer complaints

this project consists of and Jupyter notebook for developing and fine-tuning BERT model for classifying costumer complaints 
Jupyter notebook :BERT_multi_class_classification_compliants_final.ipynb

and FASTAPI application for deploying Deep learning model as a API service.
to initialise the API server. download and unzip the release artifcats https://github.com/tarunmukku/BERT-multi-class-compliants-classification/releases/tag/1.0.0
Zip File : **BERT-multi-class-compliants-classification.zip**

install requierd dependencies as per requirements.txt

Execute main.py file 

![image](https://user-images.githubusercontent.com/55400054/166152191-d3f0592c-a56f-41d6-9808-73020ecda706.png)


once executed API server would initiailised at 127.0.0.1:4000

to view swagger docs navigate to http://127.0.0.1:4000/docs#/default/model_prediction_predict_post and click on 'Try It out' button.
add text to 'user text' variable in the Request Body and click on execute
![image](https://user-images.githubusercontent.com/55400054/166152998-db7cae12-48ea-4060-8954-47e6f259c04c.png)

THe predection of DL model will be provided in the response of the service.



![image](https://user-images.githubusercontent.com/55400054/166153157-c8b86b7d-e708-41b5-ab62-5bc035fbb838.png)
