# Kube-controller-manager

## Purpose

This library contains code to expose kube-controller-manager API.


## Compatibility

There are *NO compatibility guarantees* for this repository, yet.  It is in direct support of Kubernetes, so branches
will track Kubernetes and be compatible with that repo.  As we more cleanly separate the layers, we will review the
compatibility guarantee. We have a goal to make this easier to use in the future.


## Where does it come from?

`kube-controller-manager` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/kube-controller-manager.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.


## Things you should *NOT* do

 1. Directly modify any files under `pkg` in this repo.  Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/kube-controller-manager`.
 2. Expect compatibility.  This repo is changing quickly in direct support of
    Kubernetes and the kube-controller-manager API.
