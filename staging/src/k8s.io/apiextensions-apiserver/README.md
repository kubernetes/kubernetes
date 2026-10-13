> ⚠️ **This is an automatically published [staged repository](https://git.k8s.io/kubernetes/staging#external-repository-staging-area) for Kubernetes**.   
> Contributions, including issues and pull requests, should be made to the main Kubernetes repository: [https://github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).  
> This repository is read-only for importing, and not used for direct contributions.  
> See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

# apiextensions-apiserver

Implements: https://github.com/kubernetes/design-proposals-archive/blob/main/api-machinery/thirdpartyresources.md

It provides an API for registering `CustomResourceDefinitions`.

## Purpose

This API server provides the implementation for `CustomResourceDefinitions` which is included as
delegate server inside of `kube-apiserver`.

The `apiextensions.k8s.io` API types and their generated clients are published in
[`k8s.io/apiextensions`](https://github.com/kubernetes/apiextensions), which doesn't depend on
the server libraries. The packages under `pkg/apis/apiextensions/v1`, `pkg/apis/apiextensions/v1beta1`
and `pkg/client` here are deprecated aliases of those, except that `AddToScheme` in
`pkg/apis/apiextensions/v1` and `v1beta1` also registers the conversions to the internal version.
See the [k8s.io/apiextensions README](https://github.com/kubernetes/apiextensions) for how to migrate.


## Compatibility

HEAD of this repo will match HEAD of k8s.io/apiserver, k8s.io/apimachinery, and k8s.io/client-go.

## Where does it come from?

`apiextensions-apiserver` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apiextensions-apiserver.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.

