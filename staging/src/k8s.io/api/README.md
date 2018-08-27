# Kubernetes API Schema

## Status

This repo is still in the experimental stage. 

## About

This repository contains the schema of the [Kubernetes API][api] that is served
by the [Kubernetes API Server][api-server].  

Individual APIs utilize a simple versioning strategy based on a major version
of the _specific_ API and a release of that API (e.g. `v1`, `v1alpha1`,
`v1alpha2`, `v2beta1`, etc).

Within each release, there should not be any breaking change to released
features, such as changing the type of a field type, renaming a field, or
changing a field number. Breaking changes are allowed between different
releases, such as v1alpha1 and v1alpha2.

[api]: https://kubernetes.io/docs/concepts/overview/kubernetes-api/
[api-server]: https://kubernetes.io/docs/admin/kube-apiserver/
