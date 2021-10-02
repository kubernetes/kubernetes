# Sampling New OpenAPI Data

[OpenAPI schema]: ./kubernetesapi/
[kind]: https://hub.docker.com/r/kindest/node/tags

This document describes how to fetch OpenAPI data from a
live kubernetes API server, e.g. an instance of [kind].

### Delete all currently built-in schema
```
make nuke
```

### Add a new built-in schema

In this directory, fetch the openapi schema and generate the 
corresponding swagger.go for the kubernetes api: 

```
make kubernetesapi/swagger.go
```

To fetch the schema without generating the swagger.go, you can
run:

```
make nuke
make kubernetesapi/swagger.json
```

Note that generating the swagger.go will re-fetch the schema.

You can specify a specific version with the "API_VERSION"
parameter. The default version is v1.19.1. Here is an
example for generating swagger.go for v1.14.1.

```
make kubernetesapi/swagger.go API_VERSION=v1.14.1
```

This will update the [OpenAPI schema]. The above command will
create a directory kubernetesapi/v1141 and store the resulting
swagger.json and swagger.go files there. 

### Make the schema available for use

While the above commands generate the swagger.go files, they
do not make them available for use nor do they update the
info field reported by `kustomize openapi info`. To make the
newly fetched schema and swagger.go available:

```
make kubernetesapi/openapiinfo.go
```

### Run all tests

At the top of the repository, run the tests.

```
make prow-presubmit-check >& /tmp/k.txt; echo $?
# The exit code should be zero; if not examine /tmp/k.txt
```
