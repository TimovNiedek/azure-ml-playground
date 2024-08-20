FROM mcr.microsoft.com/azureml/minimal-ubuntu22.04-py39-cpu-inference:latest

WORKDIR /app
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
