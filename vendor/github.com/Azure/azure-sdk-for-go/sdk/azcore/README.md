# Azure Core Client Module for Go

[![PkgGoDev](https://pkg.go.dev/badge/github.com/Azure/azure-sdk-for-go/sdk/azcore)](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azcore)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/go/go%20-%20azcore%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=1843&branchName=main)
[![Code Coverage](https://img.shields.io/azure-devops/coverage/azure-sdk/public/1843/main)](https://img.shields.io/azure-devops/coverage/azure-sdk/public/1843/main)

The `azcore` module provides a set of common interfaces and types for Go SDK client modules.
These modules follow the [Azure SDK Design Guidelines for Go](https://azure.github.io/azure-sdk/golang_introduction.html).

## Getting started

This project uses [Go modules](https://github.com/golang/go/wiki/Modules) for versioning and dependency management.

Typically, you will not need to explicitly install `azcore` as it will be installed as a client module dependency.
To add the latest version to your `go.mod` file, execute the following command.

```bash
go get github.com/Azure/azure-sdk-for-go/sdk/azcore
```

General documentation and examples can be found on [pkg.go.dev](https://pkg.go.dev/github.com/Azure/azure-sdk-for-go/sdk/azcore).

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
