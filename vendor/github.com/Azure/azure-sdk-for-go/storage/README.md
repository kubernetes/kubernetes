# Azure Storage SDK for Go (Preview)

:exclamation: IMPORTANT: This package is in maintenance only and will be deprecated in the
future. Please use one of the following packages instead.

| Service | Import Path/Repo |
|---------|------------------|
| Storage - Blobs | [github.com/Azure/azure-sdk-for-go/sdk/storage/azblob](https://github.com/Azure/azure-sdk-for-go/tree/main/sdk/storage/azblob) |
| Storage - Files | [github.com/Azure/azure-storage-file-go](https://github.com/Azure/azure-storage-file-go) |
| Storage - Queues | [github.com/Azure/azure-storage-queue-go](https://github.com/Azure/azure-storage-queue-go) |
| Storage - Tables | [github.com/Azure/azure-sdk-for-go/sdk/data/aztables](https://github.com/Azure/azure-sdk-for-go/tree/main/sdk/data/aztables)

The `github.com/Azure/azure-sdk-for-go/storage` package is used to manage
[Azure Storage](https://docs.microsoft.com/azure/storage/) data plane
resources: containers, blobs, tables, and queues.

To manage storage *accounts* use Azure Resource Manager (ARM) via the packages
at [github.com/Azure/azure-sdk-for-go/services/storage](https://github.com/Azure/azure-sdk-for-go/tree/main/services/storage).

This package also supports the [Azure Storage
Emulator](https://azure.microsoft.com/documentation/articles/storage-use-emulator/)
(Windows only).

