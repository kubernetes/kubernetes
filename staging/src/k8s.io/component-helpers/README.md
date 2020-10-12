# component-helpers

This repository provides helpers primarily for core components (core components as described in [Create a k8s.io/component-base repo](https://github.com/kubernetes/enhancements/blob/master/keps/sig-cluster-lifecycle/wgs/0032-create-a-k8s-io-component-repo.md#component-definition)) which are required by at least two separate binaries in kubernetes org.
Yet, still with a high level of abstraction.

`k8s.io/component-base` staging repository was considered as a candidate for hosting the helpers. Although, since the helpers are not required by the core components, the repository was deemed unsuitable.

The only allowed kubernetes dependencies are `k8s.io/apimachinery`, `k8s.io/api` and `k8s.io/client-go`.

## Purpose

One of the goals is to provide a better location for helpers currently located under `k8s.io/kubernetes/pkg/apis`.

Recent effort of moving [scheduling
 framework](https://kubernetes.io/docs/concepts/scheduling-eviction/scheduling-framework/) under
`k8s.io/kube-scheduler` requires duplication of many helper functions
(see [#91782](https://github.com/kubernetes/kubernetes/issues/91782) for more details).
Importing the helpers from this repository allows to minimize or remove already existing duplication.

Another example is shared RBAC code which is blocking extracting kubectl to staging (see https://github.com/kubernetes/enhancements/issues/1020). This problem dates all the way back to December 2018 (see SIG-CLI call from December 19, 2018: https://docs.google.com/document/d/1r0YElcXt6G5mOWxwZiXgGu_X6he3F--wKwg-9UBc29I/edit?pli=1). Recently the topic was touched during sig-auth call (see https://docs.google.com/document/d/1woLGRoONE3EBVx-wTb4pvp4CI7tmLZ6lS26VTbosLKM/edit?ts=5ef3be6a#heading=h.etc9yylhln8x).

## Compatibility

There are NO compatibility guarantees for this repository. It is in direct support of Kubernetes, so branches will track Kubernetes and be compatible with that repo. As we more cleanly separate the layers, we will review the compatibility guarantee.

## Where does it come from?

This repo is synced from https://github.com/kubernetes/kubernetes/tree/master/staging/src/k8s.io/component-helpers.
Code changes are made in that location, merged into `k8s.io/kubernetes` and later synced here by a bot.
