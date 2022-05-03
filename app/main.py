import uvicorn

from fastapi import FastAPI
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
import gdown

from pydantic import BaseModel
from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
bert = TFBertModel.from_pretrained('bert-base-uncased')
encoded_dict = ['Credit card','Savings account', 'Medical debt', 'Domestic (US) money transfer', 'Vehicle loan']
max_len = 75
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert(input_ids,attention_mask = input_mask)[0] 
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(5,activation = 'sigmoid')(out)
new_model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)


# a file

url = "https://drive.google.com/uc?id=1WjO8IZFu18L6vzx78eWF0wbDoT6ouzmo"
output = "model_weights.h5"
gdown.download(url, output, quiet=False)

class Complaint(BaseModel):
    user_text: str


app = FastAPI(title="consumer compliants classification API")
@app.get('/')

def index():

    return {'message': "This is the home page of this API."}

@app.get('/apiv1/{name}')

def greet(name: str):

    return {'message': f'Hello! @{name}'}



@app.post('/predict')
def model_prediction(data: Complaint):
  #  data.user_text
    print(data.user_text)

    x_val = tokenizer(
    text=data.user_text,
    add_special_tokens=True,
    max_length=75,
    truncation=True,
    padding='max_length', 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True) 
    validation = new_model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']}).tolist()[0]
    max_val = max(validation)
    idx_max = validation.index(max_val)

    return {'prediction': encoded_dict[idx_max]}

if __name__ == '__main__':

    uvicorn.run(app, host='127.0.0.1', port=4000, debug=True)
