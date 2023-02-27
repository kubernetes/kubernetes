# Migrating from autorest/adal to azidentity

`azidentity` provides Azure Active Directory (Azure AD) authentication for the newest Azure SDK modules (`github.com/azure-sdk-for-go/sdk/...`). Older Azure SDK packages (`github.com/azure-sdk-for-go/services/...`) use types from `github.com/go-autorest/autorest/adal` instead.

This guide shows common authentication code using `autorest/adal` and its equivalent using `azidentity`.

## Table of contents

- [Acquire a token](#acquire-a-token)
- [Client certificate authentication](#client-certificate-authentication)
- [Client secret authentication](#client-secret-authentication)
- [Configuration](#configuration)
- [Device code authentication](#device-code-authentication)
- [Managed identity](#managed-identity)
- [Use azidentity credentials with older packages](#use-azidentity-credentials-with-older-packages)

## Configuration

### `autorest/adal`

Token providers require a token audience (resource identifier) and an instance of `adal.OAuthConfig`, which requires an Azure AD endpoint and tenant:

```go
import "github.com/Azure/go-autorest/autorest/adal"

oauthCfg, err := adal.NewOAuthConfig("https://login.chinacloudapi.cn", tenantID)
handle(err)

spt, err := adal.NewServicePrincipalTokenWithSecret(
    *oauthCfg, clientID, "https://management.chinacloudapi.cn/", &adal.ServicePrincipalTokenSecret{ClientSecret: secret},
)
```

### `azidentity`

A credential instance can acquire tokens for any audience. The audience for each token is determined by the client requesting it. Credentials require endpoint configuration only for sovereign or private clouds. The `azcore/cloud` package has predefined configuration for sovereign clouds such as Azure China:

```go
import (
    "github.com/Azure/azure-sdk-for-go/sdk/azcore/cloud"
    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
)

clientOpts := azcore.ClientOptions{Cloud: cloud.AzureChina}

cred, err := azidentity.NewClientSecretCredential(
    tenantID, clientID, secret, &azidentity.ClientSecretCredentialOptions{ClientOptions: clientOpts},
)
handle(err)
```

## Client secret authentication

### `autorest/adal`

```go
import (
    "github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/subscriptions"
    "github.com/Azure/go-autorest/autorest"
    "github.com/Azure/go-autorest/autorest/adal"
)

oauthCfg, err := adal.NewOAuthConfig("https://login.microsoftonline.com", tenantID)
handle(err)
spt, err := adal.NewServicePrincipalTokenWithSecret(
    *oauthCfg, clientID, "https://management.azure.com/", &adal.ServicePrincipalTokenSecret{ClientSecret: secret},
)
handle(err)

client := subscriptions.NewClient()
client.Authorizer = autorest.NewBearerAuthorizer(spt)
```

### `azidentity`

```go
import (
    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
    "github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/resources/armsubscriptions"
)

cred, err := azidentity.NewClientSecretCredential(tenantID, clientID, secret, nil)
handle(err)

client, err := armsubscriptions.NewClient(cred, nil)
handle(err)
```

## Client certificate authentication

### `autorest/adal`

```go
import (
    "os"

    "github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/subscriptions"
    "github.com/Azure/go-autorest/autorest"
    "github.com/Azure/go-autorest/autorest/adal"
)
certData, err := os.ReadFile("./example.pfx")
handle(err)

certificate, rsaPrivateKey, err := decodePkcs12(certData, "")
handle(err)

oauthCfg, err := adal.NewOAuthConfig("https://login.microsoftonline.com", tenantID)
handle(err)

spt, err := adal.NewServicePrincipalTokenFromCertificate(
    *oauthConfig, clientID, certificate, rsaPrivateKey, "https://management.azure.com/",
)

client := subscriptions.NewClient()
client.Authorizer = autorest.NewBearerAuthorizer(spt)
```

### `azidentity`

```go
import (
    "os"

    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
    "github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/resources/armsubscriptions"
)

certData, err := os.ReadFile("./example.pfx")
handle(err)

certs, key, err := azidentity.ParseCertificates(certData, nil)
handle(err)

cred, err = azidentity.NewClientCertificateCredential(tenantID, clientID, certs, key, nil)
handle(err)

client, err := armsubscriptions.NewClient(cred, nil)
handle(err)
```

## Managed identity

### `autorest/adal`

```go
import (
    "github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/subscriptions"
    "github.com/Azure/go-autorest/autorest"
    "github.com/Azure/go-autorest/autorest/adal"
)

spt, err := adal.NewServicePrincipalTokenFromManagedIdentity("https://management.azure.com/", nil)
handle(err)

client := subscriptions.NewClient()
client.Authorizer = autorest.NewBearerAuthorizer(spt)
```

### `azidentity`

```go
import (
    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
    "github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/resources/armsubscriptions"
)

cred, err := azidentity.NewManagedIdentityCredential(nil)
handle(err)

client, err := armsubscriptions.NewClient(cred, nil)
handle(err)
```

### User-assigned identities

`autorest/adal`:

```go
import "github.com/Azure/go-autorest/autorest/adal"

opts := &adal.ManagedIdentityOptions{ClientID: "..."}
spt, err := adal.NewServicePrincipalTokenFromManagedIdentity("https://management.azure.com/")
handle(err)
```

`azidentity`:

```go
import "github.com/Azure/azure-sdk-for-go/sdk/azidentity"

opts := azidentity.ManagedIdentityCredentialOptions{ID: azidentity.ClientID("...")}
cred, err := azidentity.NewManagedIdentityCredential(&opts)
handle(err)
```

## Device code authentication

### `autorest/adal`

```go
import (
    "fmt"
    "net/http"

    "github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/subscriptions"
    "github.com/Azure/go-autorest/autorest"
    "github.com/Azure/go-autorest/autorest/adal"
)

oauthClient := &http.Client{}
oauthCfg, err := adal.NewOAuthConfig("https://login.microsoftonline.com", tenantID)
handle(err)
resource := "https://management.azure.com/"
deviceCode, err := adal.InitiateDeviceAuth(oauthClient, *oauthCfg, clientID, resource)
handle(err)

// display instructions, wait for the user to authenticate
fmt.Println(*deviceCode.Message)
token, err := adal.WaitForUserCompletion(oauthClient, deviceCode)
handle(err)

spt, err := adal.NewServicePrincipalTokenFromManualToken(*oauthCfg, clientID, resource, *token)
handle(err)

client := subscriptions.NewClient()
client.Authorizer = autorest.NewBearerAuthorizer(spt)
```

### `azidentity`

```go
import (
    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
    "github.com/Azure/azure-sdk-for-go/sdk/resourcemanager/resources/armsubscriptions"
)

cred, err := azidentity.NewDeviceCodeCredential(nil)
handle(err)

client, err := armsubscriptions.NewSubscriptionsClient(cred, nil)
handle(err)
```

`azidentity.DeviceCodeCredential` will guide a user through authentication, printing instructions to the console by default. The user prompt is customizable. For more information, see the [package documentation](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azidentity#DeviceCodeCredential).

## Acquire a token

### `autorest/adal`

```go
import "github.com/Azure/go-autorest/autorest/adal"

oauthCfg, err := adal.NewOAuthConfig("https://login.microsoftonline.com", tenantID)
handle(err)

spt, err := adal.NewServicePrincipalTokenWithSecret(
    *oauthCfg, clientID, "https://vault.azure.net", &adal.ServicePrincipalTokenSecret{ClientSecret: secret},
)

err = spt.Refresh()
if err == nil {
    token := spt.Token
}
```

### `azidentity`

In ordinary usage, application code doesn't need to request tokens from credentials directly. Azure SDK clients handle token acquisition and refreshing internally. However, applications may call `GetToken()` to do so. All credential types have this method.

```go
import (
    "github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
)

cred, err := azidentity.NewClientSecretCredential(tenantID, clientID, secret, nil)
handle(err)

tk, err := cred.GetToken(
    context.TODO(), policy.TokenRequestOptions{Scopes: []string{"https://vault.azure.net/.default"}},
)
if err == nil {
    token := tk.Token
}
```

Note that `azidentity` credentials use the Azure AD v2.0 endpoint, which requires OAuth 2 scopes instead of the resource identifiers `autorest/adal` expects. For more information, see [Azure AD documentation](https://docs.microsoft.com/azure/active-directory/develop/v2-permissions-and-consent).

## Use azidentity credentials with older packages

The [azidext module](https://pkg.go.dev/github.com/jongio/azidext/go/azidext) provides an adapter for `azidentity` credential types. The adapter enables using the credential types with older Azure SDK clients. For example:

```go
import (
    "github.com/Azure/azure-sdk-for-go/sdk/azidentity"
    "github.com/Azure/azure-sdk-for-go/services/resources/mgmt/2018-06-01/subscriptions"
    "github.com/jongio/azidext/go/azidext"
)

cred, err := azidentity.NewClientSecretCredential(tenantID, clientID, secret, nil)
handle(err)

client := subscriptions.NewClient()
client.Authorizer = azidext.NewTokenCredentialAdapter(cred, []string{"https://management.azure.com//.default"})
```

![Impressions](https://azure-sdk-impressions.azurewebsites.net/api/impressions/azure-sdk-for-go%2Fsdk%2Fazidentity%2FMIGRATION.png)
