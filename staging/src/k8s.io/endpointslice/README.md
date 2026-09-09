> ⚠️ **This is an automatically published [staged repository](https://git.k8s.io/kubernetes/staging#external-repository-staging-area) for Kubernetes**.   
> Contributions, including issues and pull requests, should be made to the main Kubernetes repository: [https://github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).  
> This repository is read-only for importing, and not used for direct contributions.  
> See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

# endpointslice

## Purpose

This repository contains packages related to the [EndpointSlices](https://github.com/kubernetes/enhancements/tree/master/keps/sig-network/0752-endpointslices)
feature.

This EndpointSlice reconciler library is not sufficiently generic to be used by
the EndpointSlice Mirroring controller. The reconciler in the EndpointSlice
mirroring controller has a 1:1 mapping between Service/Endpoints and
EndpointSlice, which results in a simpler implementation then the EndpointSlice
staging lib. Contributions to move towards the shared code being used by the
mirroring controller would be welcome.

## Compatibility

There are *NO compatibility guarantees* for this repository, yet.  It is in direct support of Kubernetes, so branches
will track Kubernetes and be compatible with that repo.

## Where does it come from?

This repository is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/endpointslice
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.

## Things you should *NOT* do

 1. Directly modify any files in this repo. Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/endpointslice`.
 2. Expect compatibility. This repo is changing quickly in direct support of Kubernetes.

### OWNERS

SIG Network owns the code.
