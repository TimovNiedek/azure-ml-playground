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
	source .env && poetry run python f1_data_predictions/run_pipeline.py train

sweep:
	echo "Starting hyperparameter sweep"
	source .env && cd f1_data_predictions && poetry run python run_pipeline.py sweep

clean:
	echo "Destroying Azure resources"
	source .env && cd configure-cli && source destroy.sh
