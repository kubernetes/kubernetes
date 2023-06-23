# endpointslice

## Purpose

This repository contains packages related to the [EndpointSlices](https://github.com/kubernetes/enhancements/tree/master/keps/sig-network/0752-endpointslices)
feature.

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
