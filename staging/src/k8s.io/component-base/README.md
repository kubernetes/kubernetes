## component-base

## Purpose

Implement KEP 32: https://github.com/kubernetes/enhancements/blob/master/keps/sig-cluster-lifecycle/wgs/0032-create-a-k8s-io-component-repo.md

The proposal is essentially about refactoring the Kubernetes core package structure in a way that all core components may share common code around:
 - ComponentConfig implementation
 - flag and command handling
 - HTTPS serving
 - delegated authn/z
 - logging.

## Compatibility

There are *NO compatibility guarantees* for this repository, yet.  It is in direct support of Kubernetes, so branches
will track Kubernetes and be compatible with that repo.  As we more cleanly separate the layers, we will review the
compatibility guarantee. We have a goal to make this easier to use in the future.


## Where does it come from?

This repository is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/component-base.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.

## Things you should *NOT* do

 1. Directly modify any files in this repo. Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/component-base`.
 2. Expect compatibility. This repo is changing quickly in direct support of Kubernetes.

### OWNERS

WG Component Standard is working on this refactoring process, which is happening incrementally, starting in the v1.14 cycle.
SIG API Machinery and SIG Cluster Lifecycle owns the code.
