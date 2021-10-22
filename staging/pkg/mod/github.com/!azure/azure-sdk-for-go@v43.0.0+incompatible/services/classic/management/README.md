# Azure Service Management packages for Go

The `github.com/Azure/azure-sdk-for-go/services/classic/management` packages are used to perform operations using the Azure Service Management (ASM), aka classic deployment model. Read more about [Azure Resource Manager vs. classic deployment](https://azure.microsoft.com/documentation/articles/resource-manager-deployment-model/). Packages for Azure Resource Manager are in the [arm](../arm) folder.
Note that this package requires Go 1.7+ to build.
This package is in mainteinance mode and will only receive bug fixes. It is recommended to [migrate to Azure Resource Manager](https://docs.microsoft.com/en-us/azure/azure-resource-manager/resource-manager-deployment-model) deployment model.

## First a Sidenote: Authentication and the Azure Service Manager

The client currently supports authentication to the Service Management
API with certificates or Azure `.publishSettings` file. You can 
download the `.publishSettings` file for your subscriptions
[here](https://manage.windowsazure.com/publishsettings).
