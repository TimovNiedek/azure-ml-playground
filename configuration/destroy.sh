az ml datastore delete --name ml_datastore

az storage container delete --name ml-datastore --account-key $AZURE_STORAGE_KEY --account-name $AZURE_STORAGE_ACCOUNT
