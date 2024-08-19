az storage container create --name ml-datastore --account-key $AZURE_STORAGE_KEY --account-name $AZURE_STORAGE_ACCOUNT

az ml datastore create --file blob_datastore.yml --set credentials.account_key=$AZURE_STORAGE_KEY --set account_name=$AZURE_STORAGE_ACCOUNT

az ml compute create -n cpu-cluster --type amlcompute --min-instances 0 --max-instances 2

az ml environment create --file conda_environment.yml
