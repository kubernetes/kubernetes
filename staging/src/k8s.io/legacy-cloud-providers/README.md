# legacy-cloud-providers

This repository hosts the legacy cloud providers that were previously hosted under
`k8s.io/kubernetes/pkg/cloudprovider/providers`. Out-of-tree cloud providers can consume
packages in this repo to support legacy implementations of their Kubernetes cloud provider.

**Note:** go-get or vendor this package as `k8s.io/legacy-cloud-providers`.

## Purpose

To be consumed by out-of-tree cloud providers that wish to support legacy behavior
from their in-tree equivalents.

## Compatibility

The legacy providers here follow the same compatibility rules as cloud providers that
were previously in `k8s.io/kubernetes/pkg/cloudprovider/providers`.

## Where does it come from?

`legacy-cloud-providers` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/legacy-cloud-providers.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.

## Things you should NOT do

 1. Add a new cloud provider here.
 2. Directly modify anything under this repo. Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/legacy-cloud-providers`.
    sig-cloudprovider.
 3. Add new features/integrations to a cloud provider in this repo. Changes sync here should only be incremental bug fixes.

