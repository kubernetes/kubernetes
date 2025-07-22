# cli-runtime

Set of helpers for creating kubectl commands, as well as kubectl plugins.


## Purpose

This library is a shared dependency for clients to work with Kubernetes API infrastructure which allows
to maintain kubectl compatible behavior.  Its first consumer is `k8s.io/kubectl`.


## Compatibility

There are *NO compatibility guarantees* for this repository.  It is in direct support of Kubernetes, so branches
will track Kubernetes and be compatible with that repo.  As we more cleanly separate the layers, we will review the
compatibility guarantee.


## Where does it come from?

`cli-runtime` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/cli-runtime.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.


## Things you should *NOT* do

 1. Add API types to this repo.  This is for the helpers, not for the types.
 2. Directly modify any files under `pkg` in this repo.  Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/cli-runtime`.
 3. Expect compatibility.  This repo is direct support of Kubernetes and the API isn't yet stable enough for API guarantees.
 4. Add any type that only makes sense only for `kubectl`.

