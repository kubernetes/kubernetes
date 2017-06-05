<!--[metadata]>
+++
title = "Storage Drivers"
description = "Explains how to use storage drivers"
keywords = ["registry, on-prem, images, tags, repository, distribution, storage drivers, advanced"]
aliases = ["/registry/storage-drivers/"]
[menu.main]
parent="smn_registry_ref"
+++
<![end-metadata]-->


# Docker Registry Storage Driver

This document describes the registry storage driver model, implementation, and explains how to contribute new storage drivers.

## Provided Drivers

This storage driver package comes bundled with several drivers:

- [inmemory](storage-drivers/inmemory.md): A temporary storage driver using a local inmemory map. This exists solely for reference and testing.
- [filesystem](storage-drivers/filesystem.md): A local storage driver configured to use a directory tree in the local filesystem.
- [s3](storage-drivers/s3.md): A driver storing objects in an Amazon Simple Storage Solution (S3) bucket.
- [azure](storage-drivers/azure.md): A driver storing objects in [Microsoft Azure Blob Storage](http://azure.microsoft.com/en-us/services/storage/).
- [swift](storage-drivers/swift.md): A driver storing objects in [Openstack Swift](http://docs.openstack.org/developer/swift/).
- [oss](storage-drivers/oss.md): A driver storing objects in [Aliyun OSS](http://www.aliyun.com/product/oss).
- [gcs](storage-drivers/gcs.md): A driver storing objects in a [Google Cloud Storage](https://cloud.google.com/storage/) bucket.

## Storage Driver API

The storage driver API is designed to model a filesystem-like key/value storage in a manner abstract enough to support a range of drivers from the local filesystem to Amazon S3 or other distributed object storage systems.

Storage drivers are required to implement the `storagedriver.StorageDriver` interface provided in `storagedriver.go`, which includes methods for reading, writing, and deleting content, as well as listing child objects of a specified prefix key.

Storage drivers are intended to be written in Go, providing compile-time
validation of the `storagedriver.StorageDriver` interface.

## Driver Selection and Configuration

The preferred method of selecting a storage driver is using the `StorageDriverFactory` interface in the `storagedriver/factory` package. These factories provide a common interface for constructing storage drivers with a parameters map. The factory model is based off of the [Register](http://golang.org/pkg/database/sql/#Register) and [Open](http://golang.org/pkg/database/sql/#Open) methods in the builtin [database/sql](http://golang.org/pkg/database/sql) package.

Storage driver factories may be registered by name using the
`factory.Register` method, and then later invoked by calling `factory.Create`
with a driver name and parameters map. If no such storage driver can be found,
`factory.Create` will return an `InvalidStorageDriverError`.

## Driver Contribution

### Writing new storage drivers

To create a valid storage driver, one must implement the
`storagedriver.StorageDriver` interface and make sure to expose this driver
via the factory system.

#### Registering

Storage drivers should call `factory.Register` with their driver name in an `init` method, allowing callers of `factory.New` to construct instances of this driver without requiring modification of imports throughout the codebase.

## Testing

Storage driver test suites are provided in
`storagedriver/testsuites/testsuites.go` and may be used for any storage
driver written in Go. Tests can be registered using the `RegisterSuite`
function, which run the same set of tests for any registered drivers.
