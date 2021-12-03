# Controller-manager

## Purpose

This library contains common code for controller managers. Principally its for
the Kube-Controller-Manager and Cloud-Controller-Manager. However other
controller managers are welcome to use this code.


## Compatibility

There are *NO compatibility guarantees* for this repository, yet.  It is in direct support of Kubernetes, so branches
will track Kubernetes and be compatible with that repo.  As we more cleanly separate the layers, we will review the
compatibility guarantee. We have a goal to make this easier to use in the future.


## Where does it come from?

This package comes from the common code between kube-controller-manager and
cloud-controller-manager. The intent is for it to contain our current
understanding of the right way to build a controller manager. There are legacy
aspects of these controller managers which should be cleaned before adding them
here.
`controller-manager` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/controller-manager.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.


## Things you should *NOT* do

 1. Directly modify any files under `pkg` in this repo.  Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/controller-manager`.
 2. Expect compatibility.  This repo is currently changing rapidly in direct support of
    Kubernetes and the controller-manager processes and the cloud provider
    extraction effort.

