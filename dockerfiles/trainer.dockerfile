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
RUN pip install . --no-cache-dir #(1)

ENTRYPOINT ["python", "-u", "BjarkeCCtemplate/models/train_model.py"]

