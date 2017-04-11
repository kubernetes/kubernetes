<!--[metadata]>
+++
title = "In-memory storage driver"
description = "Explains how to use the in-memory storage drivers"
keywords = ["registry, service, driver, images, storage,  in-memory"]
+++
<![end-metadata]-->


# In-memory storage driver (Testing Only)

For purely tests purposes, you can use the `inmemory` storage driver. This
driver is an implementation of the `storagedriver.StorageDriver` interface which
uses local memory for object storage. If you would like to run a registry from
volatile memory, use the [`filesystem` driver](filesystem.md) on a ramdisk.

**IMPORTANT**: This storage driver *does not* persist data across runs. This is why it is only suitable for testing. *Never* use this driver in production.

## Parameters

None
