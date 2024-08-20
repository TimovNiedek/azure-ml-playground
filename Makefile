.PHONY: setup build train clean all

all: build setup train

build: configuration Dockerfile
	echo "Building Docker image"
	source .env && \
	poetry export -o requirements.txt && \
	docker build --platform linux/amd64 . -t ${AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/playground/f1-training-environment && \
	docker push ${AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/playground/f1-training-environment

setup:
	echo "Setting up Azure resources"
	source .env && cd configuration && source setup.sh

train:
	echo "Starting training job"
	source .env && az ml job create --file configuration/pipeline.yml

clean:
	echo "Destroying Azure resources"
	source .env && cd configuration && source destroy.sh
