# Base image
FROM python:3.10-slim

# install python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

#COPY name_in_project name_in_image
COPY pyproject.toml pyproject.toml
COPY BjarkeCCtemplate/ BjarkeCCtemplate/

#application specific instructions
COPY requirements.txt requirements.txt
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

#installing dependencies. No chache dir to save space
WORKDIR /
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
#RUN pip install -r requirements.txt --no-cache-dir
#RUN pip install . --no-cache-dir #(1)

#what we ask the docker container to do immidiately after "run". Therefore provide train argument. Afterwards, go into vm and we can run train again.
ENTRYPOINT ["python", "-u", "BjarkeCCtemplate/train_model.py"]
