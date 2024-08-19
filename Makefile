make infra:
	echo "Setting up infrastructure"
	source .env && cd configuration && source setup.sh


make destroy:
	echo "Destroying infrastructure"
	source .env && cd configuration && source destroy.sh
