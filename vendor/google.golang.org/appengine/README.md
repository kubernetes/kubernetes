# Go App Engine packages

[![Build Status](https://travis-ci.org/golang/appengine.svg)](https://travis-ci.org/golang/appengine)

This repository supports the Go runtime on App Engine,
including both the standard App Engine and the
"App Engine flexible environment" (formerly known as "Managed VMs").
It provides APIs for interacting with App Engine services.
Its canonical import path is `google.golang.org/appengine`.

See https://cloud.google.com/appengine/docs/go/
for more information.

File issue reports and feature requests on the [Google App Engine issue
tracker](https://code.google.com/p/googleappengine/issues/entry?template=Go%20defect).

## Directory structure
The top level directory of this repository is the `appengine` package. It
contains the
basic APIs (e.g. `appengine.NewContext`) that apply across APIs. Specific API
packages are in subdirectories (e.g. `datastore`).

There is an `internal` subdirectory that contains service protocol buffers,
plus packages required for connectivity to make API calls. App Engine apps
should not directly import any package under `internal`.

## Updating a Go App Engine app

This section describes how to update an older Go App Engine app to use
these packages. A provided tool, `aefix`, can help automate steps 2 and 3
(run `go get google.golang.org/appengine/cmd/aefix` to install it), but
read the details below since `aefix` can't perform all the changes.

### 1. Update YAML files (App Engine flexible environment / Managed VMs only)

The `app.yaml` file (and YAML files for modules) should have these new lines added:
```
vm: true
```
See https://cloud.google.com/appengine/docs/go/modules/#Go_Instance_scaling_and_class for details.

### 2. Update import paths

The import paths for App Engine packages are now fully qualified, based at `google.golang.org/appengine`.
You will need to update your code to use import paths starting with that; for instance,
code importing `appengine/datastore` will now need to import `google.golang.org/appengine/datastore`.

### 3. Update code using deprecated, removed or modified APIs

Most App Engine services are available with exactly the same API.
A few APIs were cleaned up, and some are not available yet.
This list summarises the differences:

* `appengine.Context` has been replaced with the `Context` type from `golang.org/x/net/context`.
* Logging methods that were on `appengine.Context` are now functions in `google.golang.org/appengine/log`.
* `appengine.Timeout` has been removed. Use `context.WithTimeout` instead.
* `appengine.Datacenter` now takes a `context.Context` argument.
* `datastore.PropertyLoadSaver` has been simplified to use slices in place of channels.
* `delay.Call` now returns an error.
* `search.FieldLoadSaver` now handles document metadata.
* `urlfetch.Transport` no longer has a Deadline field; set a deadline on the
  `context.Context` instead.
* `aetest` no longer declares its own Context type, and uses the standard one instead.
* `taskqueue.QueueStats` no longer takes a maxTasks argument. That argument has been
  deprecated and unused for a long time.
* `appengine.BackendHostname` and `appengine.BackendInstance` were for the deprecated backends feature.
  Use `appengine.ModuleHostname`and `appengine.ModuleName` instead.
* Most of `appengine/file` and parts of `appengine/blobstore` are deprecated.
  Use [Google Cloud Storage](https://godoc.org/google.golang.org/cloud/storage) instead.
* `appengine/socket` is not required on App Engine flexible environment / Managed VMs.
  Use the standard `net` package instead.
