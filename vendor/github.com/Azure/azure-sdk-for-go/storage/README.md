# Azure Storage SDK for Go (Preview)

:exclamation: IMPORTANT: This package is in maintenance only and will be deprecated in the
future. Consider using the new package for blobs currently in preview at
[github.com/Azure/azure-storage-blob-go](https://github.com/Azure/azure-storage-blob-go).
New Table, Queue and File packages are also in development.

The `github.com/Azure/azure-sdk-for-go/storage` package is used to manage
[Azure Storage](https://docs.microsoft.com/en-us/azure/storage/) data plane
resources: containers, blobs, tables, and queues.

To manage storage *accounts* use Azure Resource Manager (ARM) via the packages
at [github.com/Azure/azure-sdk-for-go/services/storage](https://github.com/Azure/azure-sdk-for-go/tree/master/services/storage).

This package also supports the [Azure Storage
Emulator](https://azure.microsoft.com/documentation/articles/storage-use-emulator/)
(Windows only).

