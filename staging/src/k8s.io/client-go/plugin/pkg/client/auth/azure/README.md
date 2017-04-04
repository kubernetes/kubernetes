# Azure Active Directory plugin for client authentication

This plugin provides an integration with [Azure CLI](https://github.com/Azure/azure-cli) and Azure Active Directory device flow.

## Usage

### Microsoft Azure CLI 2.0

1. Configure the `apiserver` to use the Azure Active Directory as an OIDC provider with following options

   ```
   --oidc-client-id="https://management.core.windows.net/" \
   --oidc-issuer-url="https://sts.windows.net/TENANT_ID/"
   --oidc-username-claim="sub"
   ```

   * Replace `TENANT_ID` with your tenant ID.

2. Login with Azure CLI

  ```
  az login
  ```

3. Configure the `kubectl` to use the `azure` authentication provider

   ```
   kubectl config set-credentials "USER_NAME" --auth-provider=azure
   ```

   * Replace `USER_NAME` with your user name.

### Device Flow

1. Create an Azure native application following these [instructions](https://docs.microsoft.com/en-us/azure/active-directory/active-directory-app-registration)

  Assign permissions to this application to access the `https://management.core.windows.net/"` audience.

2. Configure the `apiserver` to use the Azure Active Directory as an OIDC provider with following options

   ```
   --oidc-client-id="https://management.core.windows.net/" \
   --oidc-issuer-url="https://sts.windows.net/TENANT_ID/"
   --oidc-username-claim="sub"
   ```

   * Replace `TENANT_ID` with your tenant ID.

3. Configure the `kubectl` to use the `azure` authentication provider with using the registered application

   ```
   kubectl config set-credentials "USER_NAME" --auth-provider=azure \
     --auth-provider-arg=client-id=APPLICATION_ID \
     --auth-provider-arg=tenant-id=TENANT_ID \
     --auth-provider-arg=audience=https://management.core.windows.net/
   ```

   * Replace `USER_NAME`, `APPLICATION_ID` and `TENANT_ID` with the values of the registered application.

 4. The access token is acquired when first `kubectl` command is executed

   ```
   kubeclt get pods

   To sign in, use a web browser to open the page https://aka.ms/devicelogin and enter the code DEC7D48GA to authenticate.
   ```

   * After signing in a web browser, the tokens are stored in the configuration, which will be used when executing next commands.
