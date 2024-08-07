az storage container create --name ml-datastore

az ml datastore create --file configuration/blob_datastore.yml --set credentials.account_key=$AZURE_STORAGE_ACCOUNT_KEY