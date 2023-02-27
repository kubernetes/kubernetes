# Troubleshoot Azure Identity authentication issues

This troubleshooting guide covers failure investigation techniques, common errors for the credential types in the `azidentity` module, and mitigation steps to resolve these errors.

## Table of contents

- [Handle azidentity errors](#handle-azidentity-errors)
  - [Permission issues](#permission-issues)
- [Find relevant information in errors](#find-relevant-information-in-errors)
- [Enable and configure logging](#enable-and-configure-logging)
- [Troubleshoot DefaultAzureCredential authentication issues](#troubleshoot-defaultazurecredential-authentication-issues)
- [Troubleshoot EnvironmentCredential authentication issues](#troubleshoot-environmentcredential-authentication-issues)
- [Troubleshoot ClientSecretCredential authentication issues](#troubleshoot-clientsecretcredential-authentication-issues)
- [Troubleshoot ClientCertificateCredential authentication issues](#troubleshoot-clientcertificatecredential-authentication-issues)
- [Troubleshoot UsernamePasswordCredential authentication issues](#troubleshoot-usernamepasswordcredential-authentication-issues)
- [Troubleshoot ManagedIdentityCredential authentication issues](#troubleshoot-managedidentitycredential-authentication-issues)
  - [Azure Virtual Machine managed identity](#azure-virtual-machine-managed-identity)
  - [Azure App Service and Azure Functions managed identity](#azure-app-service-and-azure-functions-managed-identity)
  - [Azure Kubernetes Service managed identity](#azure-kubernetes-service-managed-identity)
- [Troubleshoot AzureCliCredential authentication issues](#troubleshoot-azureclicredential-authentication-issues)
- [Get additional help](#get-additional-help)

## Handle azidentity errors

Any service client method that makes a request to the service may return an error due to authentication failure. This is because the credential authenticates on the first call to the service and on any subsequent call that needs to refresh an access token. Authentication errors include a description of the failure and possibly an error message from Azure Active Directory (Azure AD). Depending on the application, these errors may or may not be recoverable.

### Permission issues

Service client errors with a status code of 401 or 403 often indicate that authentication succeeded but the caller doesn't have permission to access the specified API. Check the service documentation to determine which RBAC roles are needed for the request, and ensure the authenticated user or service principal has the appropriate role assignments.

## Find relevant information in errors

Authentication errors can include responses from Azure AD and often contain information helpful in diagnosis. Consider the following error message:

```
ClientSecretCredential authentication failed
POST https://login.microsoftonline.com/3c631bb7-a9f7-4343-a5ba-a615913/oauth2/v2.0/token
--------------------------------------------------------------------------------
RESPONSE 401 Unauthorized
--------------------------------------------------------------------------------
{
  "error": "invalid_client",
  "error_description": "AADSTS7000215: Invalid client secret provided. Ensure the secret being sent in the request is the client secret value, not the client secret ID, for a secret added to app '86be4c01-505b-45e9-bfc0-9b825fd84'.\r\nTrace ID: 03da4b8e-5ffe-48ca-9754-aff4276f0100\r\nCorrelation ID: 7b12f9bb-2eef-42e3-ad75-eee69ec9088d\r\nTimestamp: 2022-03-02 18:25:26Z",
  "error_codes": [
    7000215
  ],
  "timestamp": "2022-03-02 18:25:26Z",
  "trace_id": "03da4b8e-5ffe-48ca-9754-aff4276f0100",
  "correlation_id": "7b12f9bb-2eef-42e3-ad75-eee69ec9088d",
  "error_uri": "https://login.microsoftonline.com/error?code=7000215"
}
--------------------------------------------------------------------------------
```

This error contains several pieces of information:

- __Failing Credential Type__: The type of credential that failed to authenticate. This can be helpful when diagnosing issues with chained credential types such as `DefaultAzureCredential` or `ChainedTokenCredential`.

- __Azure AD Error Code and Message__: The error code and message returned by Azure AD. This can give insight into the specific reason the request failed. For instance, in this case authentication failed because the provided client secret is incorrect. [Azure AD documentation](https://docs.microsoft.com/azure/active-directory/develop/reference-aadsts-error-codes#aadsts-error-codes) has more information on AADSTS error codes.

- __Correlation ID and Timestamp__: The correlation ID and timestamp identify the request in server-side logs. This information can be useful to support engineers diagnosing unexpected Azure AD failures.

### Enable and configure logging

`azidentity` provides the same logging capabilities as the rest of the Azure SDK. The simplest way to see the logs to help debug authentication issues is to print credential logs to the console.
```go
import azlog "github.com/Azure/azure-sdk-for-go/sdk/azcore/log"

// print log output to stdout
azlog.SetListener(func(event azlog.Event, s string) {
    fmt.Println(s)
})

// include only azidentity credential logs
azlog.SetEvents(azidentity.EventAuthentication)
```

## Troubleshoot DefaultAzureCredential authentication issues

| Error |Description| Mitigation |
|---|---|---|
|"DefaultAzureCredential failed to acquire a token"|No credential in the `DefaultAzureCredential` chain provided a token|<ul><li>[Enable logging](#enable-and-configure-logging) to get further diagnostic information.</li><li>Consult the troubleshooting guide for underlying credential types for more information.</li><ul><li>[EnvironmentCredential](#troubleshoot-environmentcredential-authentication-issues)</li><li>[ManagedIdentityCredential](#troubleshoot-visualstudiocredential-authentication-issues)</li><li>[AzureCLICredential](#troubleshoot-azureclicredential-authentication-issues)</li></ul>|
|Error from the client with a status code of 401 or 403|Authentication succeeded but the authorizing Azure service responded with a 401 (Unauthorized), or 403 (Forbidden) status code|<ul><li>[Enable logging](#enable-and-configure-logging) to determine which credential in the chain returned the authenticating token.</li><li>If an unexpected credential is returning a token, check application configuration such as environment variables.</li><li>Ensure the correct role is assigned to the authenticated identity. For example, a service specific role rather than the subscription Owner role.</li></ul>|

## Troubleshoot EnvironmentCredential authentication issues

| Error Message |Description| Mitigation |
|---|---|---|
|Missing or incomplete environment variable configuration|A valid combination of environment variables wasn't set|Ensure the appropriate environment variables are set for the intended authentication method as described in the [module documentation](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#EnvironmentCredential)|

<a id="client-secret"></a>
## Troubleshoot ClientSecretCredential authentication issues

| Error Code | Issue | Mitigation |
|---|---|---|
|AADSTS7000215|An invalid client secret was provided.|Ensure the secret provided to the credential constructor is valid. If unsure, create a new client secret using the Azure portal. Details on creating a new client secret are in [Azure AD documentation](https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal#option-2-create-a-new-application-secret).|
|AADSTS7000222|An expired client secret was provided.|Create a new client secret using the Azure portal. Details on creating a new client secret are in [Azure AD documentation](https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal#option-2-create-a-new-application-secret).|
|AADSTS700016|The specified application wasn't found in the specified tenant.|Ensure the client and tenant IDs provided to the credential constructor are correct for your application registration. For multi-tenant apps, ensure the application has been added to the desired tenant by a tenant admin. To add a new application in the desired tenant, follow the [Azure AD instructions](https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal).|

<a id="client-cert"></a>
## Troubleshoot ClientCertificateCredential authentication issues

| Error Code | Description | Mitigation |
|---|---|---|
|AADSTS700027|Client assertion contains an invalid signature.|Ensure the specified certificate has been uploaded to the application registration as described in [Azure AD documentation](https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal#option-1-upload-a-certificate).|
|AADSTS700016|The specified application wasn't found in the specified tenant.|Ensure the client and tenant IDs provided to the credential constructor are correct for your application registration. For multi-tenant apps, ensure the application has been added to the desired tenant by a tenant admin. To add a new application in the desired tenant, follow the [Azure AD instructions](https://docs.microsoft.com/azure/active-directory/develop/howto-create-service-principal-portal).|

<a id="username-password"></a>
## Troubleshoot UsernamePasswordCredential authentication issues

| Error Code | Issue | Mitigation |
|---|---|---|
|AADSTS50126|The provided username or password is invalid.|Ensure the username and password provided to the credential constructor are valid.|

<a id="managed-id"></a>
## Troubleshoot ManagedIdentityCredential authentication issues

`ManagedIdentityCredential` is designed to work on a variety of Azure hosts support managed identity. Configuration and troubleshooting vary from host to host. The below table lists the Azure hosts that can be assigned a managed identity and are supported by `ManagedIdentityCredential`.

|Host Environment| | |
|---|---|---|
|Azure Virtual Machines and Scale Sets|[Configuration](https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/qs-configure-portal-windows-vm)|[Troubleshooting](#azure-virtual-machine-managed-identity)|
|Azure App Service and Azure Functions|[Configuration](https://docs.microsoft.com/azure/app-service/overview-managed-identity)|[Troubleshooting](#azure-app-service-and-azure-functions-managed-identity)|
|Azure Kubernetes Service|[Configuration](https://azure.github.io/aad-pod-identity/docs/)|[Troubleshooting](#azure-kubernetes-service-managed-identity)|
|Azure Arc|[Configuration](https://docs.microsoft.com/azure/azure-arc/servers/managed-identity-authentication)||
|Azure Service Fabric|[Configuration](https://docs.microsoft.com/azure/service-fabric/concepts-managed-identity)||

### Azure Virtual Machine managed identity

| Error Message |Description| Mitigation |
|---|---|---|
|The requested identity hasn’t been assigned to this resource.|The IMDS endpoint responded with a status code of 400, indicating the requested identity isn’t assigned to the VM.|If using a user assigned identity, ensure the specified ID is correct.<p/><p/>If using a system assigned identity, make sure it has been enabled as described in [managed identity documentation](https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/qs-configure-portal-windows-vm#enable-system-assigned-managed-identity-on-an-existing-vm).|
|The request failed due to a gateway error.|The request to the IMDS endpoint failed due to a gateway error, 502 or 504 status code.|IMDS doesn't support requests via proxy or gateway. Disable proxies or gateways running on the VM for requests to the IMDS endpoint `http://169.254.169.254`|
|No response received from the managed identity endpoint.|No response was received for the request to IMDS or the request timed out.|<ul><li>Ensure the VM is configured for managed identity as described in [managed identity documentation](https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/qs-configure-portal-windows-vm).</li><li>Verify the IMDS endpoint is reachable on the VM. See [below](#verify-imds-is-available-on-the-vm) for instructions.</li></ul>|
|Multiple attempts failed to obtain a token from the managed identity endpoint.|The credential has exhausted its retries for a token request.|<ul><li>Refer to the error message for more details on specific failures.<li>Ensure the VM is configured for managed identity as described in [managed identity documentation](https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/qs-configure-portal-windows-vm).</li><li>Verify the IMDS endpoint is reachable on the VM. See [below](#verify-imds-is-available-on-the-vm) for instructions.</li></ul>|

#### Verify IMDS is available on the VM

If you have access to the VM, you can use `curl` to verify the managed identity endpoint is available.

```sh
curl 'http://169.254.169.254/metadata/identity/oauth2/token?resource=https://management.core.windows.net&api-version=2018-02-01' -H "Metadata: true"
```

> This command's output will contain an access token and SHOULD NOT BE SHARED, to avoid compromising account security.

### Azure App Service and Azure Functions managed identity

| Error Message |Description| Mitigation |
|---|---|---|
|Get "`http://169.254.169.254/...`" i/o timeout|The App Service host hasn't set environment variables for managed identity configuration.|<ul><li>Ensure the App Service is configured for managed identity as described in [App Service documentation](https://docs.microsoft.com/azure/app-service/overview-managed-identity).</li><li>Verify the App Service environment is properly configured and the managed identity endpoint is available. See [below](#verify-the-app-service-managed-identity-endpoint-is-available) for instructions.</li></ul>|

#### Verify the App Service managed identity endpoint is available

If you can SSH into the App Service, you can verify managed identity is available in the environment. First ensure the environment variables `IDENTITY_ENDPOINT` and `IDENTITY_SECRET` are set. Then you can verify the managed identity endpoint is available using `curl`.

```sh
curl "$IDENTITY_ENDPOINT?resource=https://management.core.windows.net&api-version=2019-08-01" -H "X-IDENTITY-HEADER: $IDENTITY_HEADER"
```

> This command's output will contain an access token and SHOULD NOT BE SHARED, to avoid compromising account security.

### Azure Kubernetes Service managed identity

#### Pod Identity

| Error Message |Description| Mitigation |
|---|---|---|
|"no azure identity found for request clientID"|The application attempted to authenticate before an identity was assigned to its pod|Verify the pod is labeled correctly. This also occurs when a correctly labeled pod authenticates before the identity is ready. To prevent initialization races, configure NMI to set the Retry-After header in its responses as described in [Pod Identity documentation](https://azure.github.io/aad-pod-identity/docs/configure/feature_flags/#set-retry-after-header-in-nmi-response).

<a id="azure-cli"></a>
## Troubleshoot AzureCliCredential authentication issues

| Error Message |Description| Mitigation |
|---|---|---|
|Azure CLI not found on path|The Azure CLI isn’t installed or isn't on the application's path.|<ul><li>Ensure the Azure CLI is installed as described in [Azure CLI documentation](https://docs.microsoft.com/cli/azure/install-azure-cli).</li><li>Validate the installation location is in the application's `PATH` environment variable.</li></ul>|
|Please run 'az login' to set up account|No account is currently logged into the Azure CLI, or the login has expired.|<ul><li>Run `az login` to log into the Azure CLI. More information about Azure CLI authentication is available in the [Azure CLI documentation](https://docs.microsoft.com/cli/azure/authenticate-azure-cli).</li><li>Verify that the Azure CLI can obtain tokens. See [below](#verify-the-azure-cli-can-obtain-tokens) for instructions.</li></ul>|

#### Verify the Azure CLI can obtain tokens

You can manually verify that the Azure CLI can authenticate and obtain tokens. First, use the `account` command to verify the logged in account.

```azurecli
az account show
```

Once you've verified the Azure CLI is using the correct account, you can validate that it's able to obtain tokens for that account.

```azurecli
az account get-access-token --output json --resource https://management.core.windows.net
```

> This command's output will contain an access token and SHOULD NOT BE SHARED, to avoid compromising account security.

## Get additional help

Additional information on ways to reach out for support can be found in [SUPPORT.md](https://github.com/Azure/azure-sdk-for-go/blob/main/SUPPORT.md).
