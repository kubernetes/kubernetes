# api

Schema of the external API types that are served by the Kubernetes API server.

## Purpose

This library is the canonical location of the Kubernetes API definition. Most likely interaction with this repository is as a dependency of client-go.

## Compatibility

Branches track Kubernetes branches and are compatible with that repo.

## Where does it come from?

`api` is synced from https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/api. Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here.

## Things you should *NOT* do

1. https://github.com/kubernetes/kubernetes/blob/master/staging/src/k8s.io/api is synced to k8s.io/api. All changes must be made in the former. The latter is read-only.
