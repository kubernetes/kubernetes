<!--[metadata]>
+++
title = "Filesystem storage driver"
description = "Explains how to use the filesystem storage drivers"
keywords = ["registry, service, driver, images, storage,  filesystem"]
+++
<![end-metadata]-->


# Filesystem storage driver

An implementation of the `storagedriver.StorageDriver` interface which uses the local filesystem.

## Parameters

`rootdirectory`: (optional) The absolute path to a root directory tree in which
to store all registry files. The registry stores all its data here so make sure
there is adequate space available. Defaults to `/var/lib/registry`.
