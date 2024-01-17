FROM python:3.9
WORKDIR /code
COPY ./website_requirement.txt /code/requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
# go to localhost:80/docs   