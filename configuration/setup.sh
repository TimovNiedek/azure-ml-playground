az storage container create --name ml-datastore

az ml datastore create --file blob_datastore.yml --set credentials.account_key=$AZURE_STORAGE_ACCOUNT_KEY

az ml datastore set-default --name ml_datastore
