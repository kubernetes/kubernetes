# apiserver

Generic library for building a Kubernetes aggregated API server.


## Purpose

This library contains code to create Kubernetes aggregation server complete with delegated authentication and authorization, 
`kubectl` compatible discovery information, optional admission chain, and versioned types.  It's first consumers are
`k8s.io/kubernetes`, `k8s.io/kube-aggregator`, and `github.com/kubernetes-incubator/service-catalog`.


## Compatibility

There are *NO compatibility guarantees* for this repository, yet.  It is in direct support of Kubernetes, so branches
will track Kubernetes and be compatible with that repo.  As we more cleanly separate the layers, we will review the
compatibility guarantee.  We have a goal to make this easier to use in 2017.


## Where does it come from?

`apiserver` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/apiserver.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.


## Things you should *NOT* do

 1. Directly modify any files under `pkg` in this repo.  Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/apiserver`.
 2. Expect compatibility.  This repo is changing quickly in direct support of
    Kubernetes and the API isn't yet stable enough for API guarantees.
