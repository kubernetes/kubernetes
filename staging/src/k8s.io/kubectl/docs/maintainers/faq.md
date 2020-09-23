# kubectl Maintainer FAQ

This document serves as a knowledge bank of SIG-Cli's stance on certain
recurring issues and historical decisions. 

## Enhancing kubectl create/run

TODO: Condense https://github.com/kubernetes/kubectl/issues/914

## kubectl get all

`kubectl get all` is a legacy command and is actually implemented with a
hardcoded server side list that is not easy to maintain. There is potential that
it will be removed in the future and therefore will not be expanded upon or
improved.

https://github.com/kubernetes/kubectl/issues/151

We recommend using [ketall](https://github.com/corneliusweig/ketall) which can
be installed standalone or via [krew](https://krew.sigs.k8s.io/).

## Confirmation when using `--all`

Introducing `--all` would be a breaking change. We’ve made an attempt in the
past (see https://github.com/kubernetes/kubernetes/pull/62167) and we’ve decided
that the suggested approach is to encourage cluster owners to implement
something like https://github.com/kubernetes/kubernetes/pull/17740 or currently
using validating webhooks (see
https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.19/#validatingwebhookconfiguration-v1-admissionregistration-k8s-io)
but no such effort will be undertaken in kubectl itself.

## Supported versions

The Kubernetes project maintains release branches for the most recent three
minor releases (1.19, 1.18, 1.17 as of this writing). [Kubernetes Version Skew
Policy](https://kubernetes.io/docs/setup/release/version-skew-policy/#supported-versions)
