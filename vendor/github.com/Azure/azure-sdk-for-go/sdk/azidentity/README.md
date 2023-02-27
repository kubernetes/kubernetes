# Azure Identity Client Module for Go

The Azure Identity module provides Azure Active Directory (Azure AD) token authentication support across the Azure SDK. It includes a set of `TokenCredential` implementations, which can be used with Azure SDK clients supporting token authentication.

[![PkgGoDev](https://pkg.go.dev/badge/github.com/Azure/azure-sdk-for-go/sdk/azidentity)](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity)
| [Azure Active Directory documentation](https://docs.microsoft.com/azure/active-directory/)
| [Source code](https://github.com/Azure/azure-sdk-for-go/tree/main/sdk/azidentity)

# Getting started

## Install the module

This project uses [Go modules](https://github.com/golang/go/wiki/Modules) for versioning and dependency management.

Install the Azure Identity module:

```sh
go get -u github.com/Azure/azure-sdk-for-go/sdk/azidentity
```

## Prerequisites

- an [Azure subscription](https://azure.microsoft.com/free/)
- Go 1.18

### Authenticating during local development

When debugging and executing code locally, developers typically use their own accounts to authenticate calls to Azure services. The `azidentity` module supports authenticating through developer tools to simplify local development.

#### Authenticating via the Azure CLI

`DefaultAzureCredential` and `AzureCLICredential` can authenticate as the user
signed in to the [Azure CLI](https://docs.microsoft.com/cli/azure). To sign in to the Azure CLI, run `az login`. On a system with a default web browser, the Azure CLI will launch the browser to authenticate a user.

When no default browser is available, `az login` will use the device code
authentication flow. This can also be selected manually by running `az login --use-device-code`.

## Key concepts

### Credentials

A credential is a type which contains or can obtain the data needed for a
service client to authenticate requests. Service clients across the Azure SDK
accept a credential instance when they are constructed, and use that credential
to authenticate requests.

The `azidentity` module focuses on OAuth authentication with Azure Active
Directory (AAD). It offers a variety of credential types capable of acquiring
an Azure AD access token. See [Credential Types](#credential-types "Credential Types") for a list of this module's credential types.

### DefaultAzureCredential

`DefaultAzureCredential` is appropriate for most apps that will be deployed to Azure. It combines common production credentials with development credentials. It attempts to authenticate via the following mechanisms in this order, stopping when one succeeds:

![DefaultAzureCredential authentication flow](img/mermaidjs/DefaultAzureCredentialAuthFlow.svg)

1. **Environment** - `DefaultAzureCredential` will read account information specified via [environment variables](#environment-variables) and use it to authenticate.
2. **Managed Identity** - If the app is deployed to an Azure host with managed identity enabled, `DefaultAzureCredential` will authenticate with it.
3. **Azure CLI** - If a user or service principal has authenticated via the Azure CLI `az login` command, `DefaultAzureCredential` will authenticate that identity.

> Note: `DefaultAzureCredential` is intended to simplify getting started with the SDK by handling common scenarios with reasonable default behaviors. Developers who want more control or whose scenario isn't served by the default settings should use other credential types.

## Managed Identity

`DefaultAzureCredential` and `ManagedIdentityCredential` support
[managed identity authentication](https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/overview)
in any hosting environment which supports managed identities, such as (this list is not exhaustive):
* [Azure App Service](https://docs.microsoft.com/azure/app-service/overview-managed-identity)
* [Azure Arc](https://docs.microsoft.com/azure/azure-arc/servers/managed-identity-authentication)
* [Azure Cloud Shell](https://docs.microsoft.com/azure/cloud-shell/msi-authorization)
* [Azure Kubernetes Service](https://docs.microsoft.com/azure/aks/use-managed-identity)
* [Azure Service Fabric](https://docs.microsoft.com/azure/service-fabric/concepts-managed-identity)
* [Azure Virtual Machines](https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/how-to-use-vm-token)

## Examples

- [Authenticate with DefaultAzureCredential](#authenticate-with-defaultazurecredential "Authenticate with DefaultAzureCredential")
- [Define a custom authentication flow with ChainedTokenCredential](#define-a-custom-authentication-flow-with-chainedtokencredential "Define a custom authentication flow with ChainedTokenCredential")
- [Specify a user-assigned managed identity for DefaultAzureCredential](#specify-a-user-assigned-managed-identity-for-defaultazurecredential)

### Authenticate with DefaultAzureCredential

This example demonstrates authenticating a client from the `armresources` module with `DefaultAzureCredential`.

```go
cred, err := azidentity.NewDefaultAzureCredential(nil)
if err != nil {
  // handle error
}

client := armresources.NewResourceGroupsClient("subscription ID", cred, nil)
```

### Specify a user-assigned managed identity for DefaultAzureCredential

To configure `DefaultAzureCredential` to authenticate a user-assigned managed identity, set the environment variable `AZURE_CLIENT_ID` to the identity's client ID.

### Define a custom authentication flow with `ChainedTokenCredential`

`DefaultAzureCredential` is generally the quickest way to get started developing apps for Azure. For more advanced scenarios, `ChainedTokenCredential` links multiple credential instances to be tried sequentially when authenticating. It will try each chained credential in turn until one provides a token or fails to authenticate due to an error.

The following example demonstrates creating a credential, which will attempt to authenticate using managed identity. It will fall back to authenticating via the Azure CLI when a managed identity is unavailable.

```go
managed, err := azidentity.NewManagedIdentityCredential(nil)
if err != nil {
  // handle error
}
azCLI, err := azidentity.NewAzureCLICredential(nil)
if err != nil {
  // handle error
}
chain, err := azidentity.NewChainedTokenCredential([]azcore.TokenCredential{managed, azCLI}, nil)
if err != nil {
  // handle error
}

client := armresources.NewResourceGroupsClient("subscription ID", chain, nil)
```

## Credential Types

### Authenticating Azure Hosted Applications

|Credential|Usage
|-|-
|[DefaultAzureCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#DefaultAzureCredential)|Simplified authentication experience for getting started developing Azure apps
|[ChainedTokenCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#ChainedTokenCredential)|Define custom authentication flows, composing multiple credentials
|[EnvironmentCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#EnvironmentCredential)|Authenticate a service principal or user configured by environment variables
|[ManagedIdentityCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#ManagedIdentityCredential)|Authenticate the managed identity of an Azure resource

### Authenticating Service Principals

|Credential|Usage
|-|-
|[ClientAssertionCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity@v1.2.0-beta.2#ClientAssertionCredential)|Authenticate a service principal with a signed client assertion
|[ClientCertificateCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#ClientCertificateCredential)|Authenticate a service principal with a certificate
|[ClientSecretCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#ClientSecretCredential)|Authenticate a service principal with a secret

### Authenticating Users

|Credential|Usage
|-|-
|[InteractiveBrowserCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#InteractiveBrowserCredential)|Interactively authenticate a user with the default web browser
|[DeviceCodeCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#DeviceCodeCredential)|Interactively authenticate a user on a device with limited UI
|[UsernamePasswordCredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#UsernamePasswordCredential)|Authenticate a user with a username and password

### Authenticating via Development Tools

|Credential|Usage
|-|-
|[AzureCLICredential](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#AzureCLICredential)|Authenticate as the user signed in to the Azure CLI

## Environment Variables

`DefaultAzureCredential` and `EnvironmentCredential` can be configured with environment variables. Each type of authentication requires values for specific variables:

#### Service principal with secret

|variable name|value
|-|-
|`AZURE_CLIENT_ID`|ID of an Azure Active Directory application
|`AZURE_TENANT_ID`|ID of the application's Azure Active Directory tenant
|`AZURE_CLIENT_SECRET`|one of the application's client secrets

#### Service principal with certificate

|variable name|value
|-|-
|`AZURE_CLIENT_ID`|ID of an Azure Active Directory application
|`AZURE_TENANT_ID`|ID of the application's Azure Active Directory tenant
|`AZURE_CLIENT_CERTIFICATE_PATH`|path to a certificate file including private key
|`AZURE_CLIENT_CERTIFICATE_PASSWORD`|password of the certificate file, if any

#### Username and password

|variable name|value
|-|-
|`AZURE_CLIENT_ID`|ID of an Azure Active Directory application
|`AZURE_USERNAME`|a username (usually an email address)
|`AZURE_PASSWORD`|that user's password

Configuration is attempted in the above order. For example, if values for a
client secret and certificate are both present, the client secret will be used.

## Troubleshooting

### Error Handling

Credentials return an `error` when they fail to authenticate or lack data they require to authenticate. For guidance on resolving errors from specific credential types, see the [troubleshooting guide](https://aka.ms/azsdk/go/identity/troubleshoot).

For more details on handling specific Azure Active Directory errors please refer to the
Azure Active Directory
[error code documentation](https://docs.microsoft.com/azure/active-directory/develop/reference-aadsts-error-codes).

### Logging

This module uses the classification-based logging implementation in `azcore`. To enable console logging for all SDK modules, set `AZURE_SDK_GO_LOGGING` to `all`. Use the `azcore/log` package to control log event output or to enable logs for `azidentity` only. For example:
```go
import azlog "github.com/Azure/azure-sdk-for-go/sdk/azcore/log"

// print log output to stdout
azlog.SetListener(func(event azlog.Event, s string) {
    fmt.Println(s)
})

// include only azidentity credential logs
azlog.SetEvents(azidentity.EventAuthentication)
```

Credentials log basic information only, such as `GetToken` success or failure and errors. These log entries don't contain authentication secrets but may contain sensitive information.

## Next steps

Client and management modules listed on the [Azure SDK releases page](https://azure.github.io/azure-sdk/releases/latest/go.html) support authenticating with `azidentity` credential types. You can learn more about using these libraries in their documentation, which is linked from the release page.

## Provide Feedback

If you encounter bugs or have suggestions, please
[open an issue](https://github.com/Azure/azure-sdk-for-go/issues).

## Contributing

This project welcomes contributions and suggestions. Most contributions require
you to agree to a Contributor License Agreement (CLA) declaring that you have
the right to, and actually do, grant us the rights to use your contribution.
For details, visit [https://cla.microsoft.com](https://cla.microsoft.com).

When you submit a pull request, a CLA-bot will automatically determine whether
you need to provide a CLA and decorate the PR appropriately (e.g., label,
comment). Simply follow the instructions provided by the bot. You will only
need to do this once across all repos using our CLA.

This project has adopted the
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the
[Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any
additional questions or comments.

![Impressions](https://azure-sdk-impressions.azurewebsites.net/api/impressions/azure-sdk-for-go%2Fsdk%2Fazidentity%2FREADME.png)
