> ⚠️ **This is an automatically published [staged repository](https://git.k8s.io/kubernetes/staging#external-repository-staging-area) for Kubernetes**.   
> Contributions, including issues and pull requests, should be made to the main Kubernetes repository: [https://github.com/kubernetes/kubernetes](https://github.com/kubernetes/kubernetes).  
> This repository is read-only for importing, and not used for direct contributions.  
> See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.

# cluster-bootstrap

Set of constants and helpers in support of
https://github.com/kubernetes/design-proposals-archive/blob/main/cluster-lifecycle/bootstrap-discovery.md


## Purpose

Current user is kubeadm, the controller that cleans up the tokens, and the bootstrap authenticator.


## Where does it come from?

`cluster-bootstrap` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/cluster-bootstrap.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.


## Things you should *NOT* do

 1. Add API types to this repo.  This is for the helpers, not for the types.
 2. Directly modify any files under `token` in this repo.  Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/cluster-bootstrap`.

