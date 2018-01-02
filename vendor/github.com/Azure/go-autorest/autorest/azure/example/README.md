# autorest azure example

## Usage (device mode)

This shows how to use the example for device auth.

1. Execute this. It will save your token to /tmp/azure-example-token:

    ```
    ./example -tenantId "13de0a15-b5db-44b9-b682-b4ba82afbd29" -subscriptionId "aff271ee-e9be-4441-b9bb-42f5af4cbaeb" -mode "device" -tokenCachePath "/tmp/azure-example-token"
    ```

2. Execute it again, it will load the token from cache and not prompt for auth again.

## Usage (certificate mode)

This example covers how to make an authenticated call to the Azure Resource Manager APIs, using certificate-based authentication.

0. Export some required variables

    ```
    export SUBSCRIPTION_ID="aff271ee-e9be-4441-b9bb-42f5af4cbaeb"
    export TENANT_ID="13de0a15-b5db-44b9-b682-b4ba82afbd29"
    export RESOURCE_GROUP="someresourcegroup"
    ```

    * replace both values with your own

1. Create a private key

    ```
    openssl genrsa -out "example.key" 2048
    ```



2. Create the certificate

    ```
    openssl req -new -key "example.key" -subj "/CN=example" -out "example.csr"

    openssl x509 -req -in "example.csr" -signkey "example.key" -out "example.crt" -days 10000
    ```



3. Create the PKCS12 version of the certificate (with no password)

    ```
    openssl pkcs12 -export -out "example.pfx" -inkey "example.key" -in "example.crt" -passout pass:
    ```



4. Register a new Azure AD Application with the certificate contents

    ```
    certificateContents="$(tail -n+2 "example.key" | head -n-1)"
   
    azure ad app create \
        --name "example-azuread-app" \
        --home-page="http://example-azuread-app/home" \
        --identifier-uris "http://example-azuread-app/app" \
        --key-usage "Verify" \
        --end-date "2020-01-01" \
        --key-value "${certificateContents}"
    ```



5. Create a new service principal using the "Application Id" from the previous step

    ```
    azure ad sp create "APPLICATION_ID"
    ```

    * Replace APPLICATION_ID with the "Application Id" returned in step 4



6. Grant your service principal necessary permissions

    ```
    azure role assignment create \
        --resource-group "${RESOURCE_GROUP}" \
        --roleName "Contributor" \
        --subscription "${SUBSCRIPTION_ID}" \
        --spn "http://example-azuread-app/app"
    ```

    * Replace SUBSCRIPTION_ID with your subscription id
    * Replace RESOURCE_GROUP with the resource group for the assignment
    * Ensure that the `spn` parameter matches an `identifier-url` from Step 4



7. Run this example app to see your resource groups

    ```
    go run main.go \
        --tenantId="${TENANT_ID}" \
        --subscriptionId="${SUBSCRIPTION_ID}" \
        --applicationId="http://example-azuread-app/app" \
        --certificatePath="certificate.pfx"
    ```


You should see something like this as output:

```
2015/11/08 18:28:39 Using these settings:
2015/11/08 18:28:39 * certificatePath: certificate.pfx
2015/11/08 18:28:39 * applicationID: http://example-azuread-app/app
2015/11/08 18:28:39 * tenantID: 13de0a15-b5db-44b9-b682-b4ba82afbd29
2015/11/08 18:28:39 * subscriptionID: aff271ee-e9be-4441-b9bb-42f5af4cbaeb
2015/11/08 18:28:39 loading certificate... 
2015/11/08 18:28:39 retrieve oauth token... 
2015/11/08 18:28:39 querying the list of resource groups... 
2015/11/08 18:28:50 
2015/11/08 18:28:50 Groups: {"value":[{"id":"/subscriptions/aff271ee-e9be-4441-b9bb-42f5af4cbaeb/resourceGroups/kube-66f30810","name":"kube-66f30810","location":"westus","tags":{},"properties":{"provisioningState":"Succeeded"}}]}
```



## Notes

You may need to wait sometime between executing step 4, step 5 and step 6. If you issue those requests too quickly, you might hit an AD server that is not consistent with the server where the resource was created.
