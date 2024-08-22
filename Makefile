.PHONY: setup build train clean all

all: build setup train

build: configure-cli Dockerfile
	echo "Building Docker image"
	source .env && \
	poetry export -o requirements.txt && \
	docker build --platform linux/amd64 . -t ${AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/playground/f1-training-environment && \
	docker push ${AZURE_CONTAINER_REGISTRY_NAME}.azurecr.io/playground/f1-training-environment

setup:
	echo "Setting up Azure resources"
	source .env && cd configure-cli && source setup.sh

train:
	echo "Starting training job"
	source .env && az ml job create --file f1_data_predictions/pipeline.yml

clean:
	echo "Destroying Azure resources"
	source .env && cd configure-cli && source destroy.sh
