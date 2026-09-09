# Sampling New OpenAPI Data

[OpenAPI schema]: ./kubernetesapi/
[Kustomization schema]: ./kustomizationapi/
[kind]: https://hub.docker.com/r/kindest/node/tags

This document describes how to fetch OpenAPI data from a
live kubernetes API server. 
The scripts used will create a clean [kind] instance for this purpose.

## Replacing the default openapi schema version

### Delete all currently built-in schema

This will remove both the Kustomization and Kubernetes schemas:

```
make nuke
```

### Choose the new version to use

The compiled-in schema version should maximize API availability with respect to all actively supported Kubernetes versions. For example, while 1.20, 1.21 and 1.22 are the actively supported versions, 1.21 is the best choice. This is because 1.21 introduces at least one new API and does not remove any, while 1.22 removes a large set of long-deprecated APIs that are still supported in 1.20/1.21.

### Generating additional schema

If you'd like to change the default schema version, then in the Makefile in this directory, update the `API_VERSION` to your desired version.

You may need to update the version of Kind these scripts use by changing `KIND_VERSION` in the Makefile in this directory. You can find compatibility information in the [kind release notes](https://github.com/kubernetes-sigs/kind/releases).

In this directory, fetch the openapi schema, generate the
corresponding swagger.go for the kubernetes api, and update `kubernetesapi/openapiinfo.go`:

```
make all
```

If you want to run the steps individually instead of using `make all`, you can run
the following commands:

```
make kustomizationapi/swagger.go
make kubernetesapi/swagger.go
make kubernetesapi/openapiinfo.go
```

You can optionally delete the old `swagger.pb` and `swagger.go` files if we no longer need to support that kubernetes version of
openapi data. Make sure you rerun `make kubernetesapi/openapiinfo.go` after deleting any old schemas.


#### Precomputations

To avoid expensive schema lookups, some functions have precomputed results based on the schema. Unit tests
ensure these are kept in sync with the schema; if these tests fail you will need to follow the suggested diff
to update the precomputed results.

### Run all tests

At the top of the repository, run the tests.

```
make prow-presubmit-check >& /tmp/k.txt; echo $?
```

The exit code should be zero; if not, examine `/tmp/k.txt`.

## Partial regeneration

You can also regenerate the kubernetes api schemas specifically with:

```
rm kubernetesapi/swagger.go
make kubernetesapi/swagger.go
```

To fetch the schema without generating the swagger.go, you can
run:

```
rm kubernetesapi/swagger.pb
make kubernetesapi/swagger.pb
```

Note that generating the swagger.go will re-fetch the schema.
