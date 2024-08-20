.PHONY: setup build train clean all

all: build setup train

build:
	echo "Building Docker image"
	source .env && \
	poetry export -o requirements.txt && \
	docker build --platform linux/amd64 . -t ${AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/playground/f1-training-environment && \
	docker push ${AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/playground/f1-training-environment

setup: build
	echo "Setting up Azure resources"
	source .env && cd configuration && source setup.sh

train: build
	echo "Starting training job"
	source .env && az ml job create --file configuration/pipeline.yml

clean:
	echo "Destroying Azure resources"
	source .env && cd configuration && source destroy.sh
