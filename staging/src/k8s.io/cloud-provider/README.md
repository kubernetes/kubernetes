# cloud-provider

This repository defines the cloud-provider interface and mechanism to initialize
a cloud-provider implementation into Kubernetes. Currently multiple processes
use this code although the intent is that it will eventually only be cloud
controller manager.

**Note:** go-get or vendor this package as `k8s.io/cloud-provider`.

## Purpose

This library is a shared dependency for processes which need to be able to
integrate with cloud-provider specific functionality.

## Compatibility

Cloud Providers are expected to keep the HEAD of their implementations in sync
with the HEAD of this repository.

## Where does it come from?

`cloud-provider` is synced from
https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/cloud-provider.
Code changes are made in that location, merged into k8s.io/kubernetes and
later synced here.

## Things you should NOT do

 1. Add an cloud provider specific code to this repo.
 2. Directly modify anything under vendor/k8s.io/cloud-provider in this repo. Those are driven from `k8s.io/kubernetes/staging/src/k8s.io/cloud-provider`.
 3. Make interface changes without first discussing them with
    sig-cloudprovider.

