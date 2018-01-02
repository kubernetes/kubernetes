# Example Accessing the Azure DNS API

## Prerequisites

1. Create an Azure Resource Group. The code assumes `delete-dns` as the name: `az group create -l westus -n delete-dns`

2. Create a Service Principal to access the resource group for example with the Azure CLI
```
az ad sp create-for-rbac --role contributor --scopes /subscriptions/<your subscription id>/resourceGroups/delete-dns

{
  "appId": "<appId>",
  "displayName": "<displayName>",
  "name": "<name>",
  "password": "<password>",
  "tenant": "<tenantId>"
}
```
3. Set the following environment variables from the service principal
- AZURE_CLIENT_ID=<appId>
- AZURE_CLIENT_SECRET=<password>
- AZURE_SUBSCRIPTION_ID=<your subscrption id>
- AZURE_TENANT_ID=<tenantId>

You can query the subscription id for your subscription with: `az account show --query id`

4. Get the dependencies
- go get github.com/Azure/go-autorest/autorest
- go get github.com/Azure/go-autorest/autorest/azure
- github.com/Azure/azure-sdk-for-go/arm

## Run the sample

Execute with `go run create.go`