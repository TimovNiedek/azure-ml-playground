# Azure ML Playground

## Prerequisites

* Set environment variables `AZURE_STORAGE_CONNECTION_STRING`, `AZURE_STORAGE_ACCOUNT` and `AZURE_STORAGE_ACCOUNT_KEY`.
* Run `az login`
* Run `az configure --defaults group=<resource_group>` and `az configure --defaults workspace=<ML workspace>`

Create a .env file with the following content:

```
AZURE_STORAGE_CONNECTION_STRING="..."
AZURE_STORAGE_ACCOUNT="..."
AZURE_STORAGE_KEY="..."
AZURE_SUBSCRIPTION_ID="..."
AZURE_RESOURCE_GROUP="..."
AZURE_ML_WORKSPACE="..."
AZURE_CONTAINER_REGISTRY_NAME="..."
```
