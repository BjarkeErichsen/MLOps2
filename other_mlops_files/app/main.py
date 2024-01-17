"""
Creating an API

    1  cd into repo
    2  uvicorn --reload --port 8000 main:app
"""
from http import HTTPStatus
from fastapi import FastAPI
from enum import Enum
from fastapi import UploadFile, File
from typing import Optional
import re
import cv2
from fastapi.responses import FileResponse


app = FastAPI()

class ItemEnum(Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


@app.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
"""
you provide an item through the GET request.
The decorator picks it up, and provides it as an input to the function.
The function returns it, and it shows up in the respose.
"""
@app.get("/restric_items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id}

database = {'username': [ ], 'password': [ ]}

@app.post("/login/")
def login(username: str, password: str):
    username_db = database['username']
    password_db = database['password']
    if username not in username_db and password not in password_db:
        with open('database.csv', "a") as file:
            file.write(f"{username}, {password} \n")
        username_db.append(username)
        password_db.append(password)
    return "login saved"

@app.get("/text_model/")
def contains_email(data: str):
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    response = {
        "input": data,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "is_email": re.fullmatch(regex, data) is not None
    }
    return response


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...), h: int = 500, w: int = 500):
    with open('image.jpg', 'wb') as image:
        content = await data.read()
        image.write(content)
        image.close()
    
    img = cv2.imread("image.jpg")
    res = cv2.resize(img, (h, w))
    cv2.imwrite('image_resize.jpg', res)
    return FileResponse('image_resize.jpg')



