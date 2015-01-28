# Go App Engine for Managed VMs

[![Build Status](https://travis-ci.org/golang/appengine.svg)](https://travis-ci.org/golang/appengine)

This repository supports the Go runtime for Managed VMs on App Engine.
It provides APIs for interacting with App Engine services.
Its canonical import path is `google.golang.org/appengine`.

See https://cloud.google.com/appengine/docs/go/managed-vms/
for more information.

## Directory structure
The top level directory of this repository is the `appengine` package. It
contains the
basic types (e.g. `appengine.Context`) that are used across APIs. Specific API
packages are in subdirectories (e.g. `datastore`).

There is an `internal` subdirectory that contains service protocol buffers,
plus packages required for connectivity to make API calls. App Engine apps
should not directly import any package under `internal`.

## Updating a Go App Engine app

This section describes how to update a traditional Go App Engine app to run on Managed VMs.

### 1. Update YAML files

The `app.yaml` file (and YAML files for modules) should have these new lines added:
```
vm: true
manual_scaling:
  instances: 1
```
See https://cloud.google.com/appengine/docs/go/modules/#Go_Instance_scaling_and_class for details.

### 2. Update import paths

The import paths for App Engine packages are now fully qualified, based at `google.golang.org/appengine`.
You will need to update your code to use import paths starting with that; for instance,
code importing `appengine/datastore` will now need to import `google.golang.org/appengine/datastore`.
You can do that manually, or by running this command to recursively update all Go source files in the current directory:
(may require GNU sed)
```
sed -i '/"appengine/{s,"appengine,"google.golang.org/appengine,;s,appengine_,appengine/,}' \
  $(find . -name '*.go')
```

### 3. Update code using deprecated, removed or modified APIs

Most App Engine services are available with exactly the same API.
A few APIs were cleaned up, and some are not available yet.
This list summarises the differences:

* `appengine.Datacenter` now takes an `appengine.Context` argument.
* `datastore.PropertyLoadSaver` has been simplified to use slices in place of channels.
* `search.FieldLoadSaver` now handles document metadata.
* `taskqueue.QueueStats` no longer takes a maxTasks argument. That argument has been
  deprecated and unused for a long time.
* `appengine/aetest`, `appengine/blobstore`, `appengine/cloudsql`
  and `appengine/runtime` have not been ported yet.
* `appengine.BackendHostname` and `appengine.BackendInstance` were for the deprecated backends feature.
  Use `appengine.ModuleHostname`and `appengine.ModuleName` instead.
* `appengine.IsCapabilityDisabled` and `appengine/capability` are obsolete.
* Most of `appengine/file` is deprecated. Use [Google Cloud Storage](https://godoc.org/google.golang.org/cloud/storage) instead.
* `appengine/socket` is deprecated. Use the standard `net` package instead.
