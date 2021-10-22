# Generate code

## Generate SDK packages

### Generate an Azure-SDK-for-Go service package

1. [Install AutoRest](https://github.com/Azure/autorest#installing-autorest).

1. Call autorest with the following arguments...

``` cmd
autorest path/to/readme/file --go --go-sdk-folder=<your/gopath/src/github.com/Azure/azure-sdk-for-go> --package-version=<version> --user-agent=<Azure-SDK-For-Go/version services> [--tag=choose/a/tag/in/the/readme/file]
```

For example...

``` cmd
autorest C:/azure-rest-api-specs/specification/advisor/resource-manager/readme.md --go --go-sdk-folder=C:/goWorkspace/src/github.com/Azure/azure-sdk-for-go --tag=package-2016-07-preview --package-version=v11.2.0-beta --user-agent='Azure-SDK-For-Go/v11.2.0-beta services'
```

- If you are looking to generate code based on a specific swagger file, you can replace `path/to/readme/file` with `--input-file=path/to/swagger/file`.
- If the readme file you want to use as input does not have golang tags yet, you can call autorest like this...

``` cmd
autorest path/to/readme/file --go --license-header=<MICROSOFT_APACHE_NO_VERSION> --namespace=<packageName> --output-folder=<your/gopath/src/github.com/Azure/azure-sdk-for-go/services/serviceName/mgmt/APIversion/packageName> --package-version=<version> --user-agent=<Azure-SDK-For-Go/version services> --clear-output-folder --can-clear-output-folder --tag=<choose/a/tag/in/the/readme/file>
```

For example...

``` cmd
autorest --input-file=https://raw.githubusercontent.com/Azure/azure-rest-api-specs/current/specification/network/resource-manager/Microsoft.Network/2017-10-01/loadBalancer.json --go --license-header=MICROSOFT_APACHE_NO_VERSION --namespace=lb --output-folder=C:/goWorkspace/src/github.com/Azure/azure-sdk-for-go/services/network/mgmt/2017-09-01/network/lb --package-version=v11.2.0-beta --clear-output-folder --can-clear-output-folder
```

1. Run `go fmt` on the generated package folder.

1. To make sure the SDK has been generated correctly, also run `golint`, `go build` and `go vet`.

### Generate Azure SDK for Go service packages in bulk

All services, all API versions.

1. [Install AutoRest](https://github.com/Azure/autorest#installing-autorest).

This repo contains a tool to generate the SDK, which depends on the golang tags from the readme files in the Azure REST API specs repo. The tool assumes you have an [Azure REST API specs](https://github.com/Azure/azure-rest-api-specs) clone, and [golint](https://github.com/golang/lint) is installed.

1. `cd tools/generator`

1. `go install`

1. Add `GOPATH/bin` to your `PATH`, in case it was not already there.

1. Call the generator tool like this...

``` cmd
generator –r [–v] [–l=logs/output/folder] –version=<version> path/to/your/swagger/repo/clone
```

For example...

``` cmd
generator –r –v –l=temp –version=v11.2.0-beta C:/azure-rest-api-specs
```

The generator tool already runs `go fmt`, `golint`, `go build` and `go vet`; so running them is not necessary.

#### Use the generator tool to generate a single package

1. Just call the generator tool specifying the service to be generated in the input folder.

``` cmd
generator –r [–v] [–l=logs/output/folder] –version=<version> path/to/your/swagger/repo/clone/specification/service
```

For example...

``` cmd
generator –r –v –l=temp –version=v11.2.0-beta C:/azure-rest-api-specs/specification/network
```

## Include a new package in the SDK

1. Submit a pull request to the Azure REST API specs repo adding the golang tags for the service and API versions in the service readme file, if the needed tags are not there yet.

1. Once the tags are available in the Azure REST API specs repo, generate the SDK.

1. In the changelog file, document the new generated SDK. Include the [autorest.go extension](https://github.com/Azure/autorest.go) version used, and the Azure REST API specs repo commit from where the SDK was generated.

1. Install [dep](https://github.com/golang/dep).

1. Run `dep ensure`.

1. Submit a pull request to this repo, and we will review it.

## Generate Azure SDK for Go profiles

Take a look into the [profile generator documentation](tools/profileBuilder)
