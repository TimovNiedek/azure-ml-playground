make setup:
	echo "Setting up Azure resources"
	source .env && cd configuration && source setup.sh


make destroy:
	echo "Destroying Azure resources"
	source .env && cd configuration && source destroy.sh

make train:
	echo "Starting training job"
	source .env && az ml job create --file configuration/pipeline.yml
