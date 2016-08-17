# Enable purging resources in `kubectl apply`

## Motivation

See [#19805](https://github.com/kubernetes/kubernetes/issues/19805).

Currently `kubectl apply` will create any new resources that exist in the resource file(s) it is supplied with, but it has no way of deleting anything that has been removed from such files.

This makes it difficult to use `kubectl apply` to update reality to match the declarative definitions of resources when that definition evolves over time, since changes can include deletions.

The primary driver for this use-case is to be able to manage cluster add-ons using `kubectl apply`.

## Proposed Solution

Use labels as a de-facto way of grouping resources for the purpose of _purging_ (removing) resources which no longer exist.

Introduce an optional `--purge-missing-where <label-selector>` flag to `kubectl apply`, which, as well as creating and updating API objects (as `kubectl apply` already does) will also purge (delete) resources which:

* match the given label selector
* do not exist in the provided files
* were created with `kubectl apply` (means: have annotation `kubectl.kubernetes.io/last-applied-configuration`)

## Managing Add-ons

Let's consider the add-ons use case.

For example, it should be possible to install and upgrade Weave Net on Kubernetes with:

```
kubectl apply -f --purge-missing-where 'weave-net' \
    https://raw.githubusercontent.com/weaveworks/weave-kube/master/weave-daemonset.yaml
```

Observe that the resulting API objects have label `weave-net`.

If `weave-daemonset.yaml` later changes and some of the API objects are removed, re-running this command will be sufficient to purge (clean up) any left-over resources.

## Managing Any Other Application

It should be possible to apply the same approach to any other applications, with the assumption that a single application can be uniquely labelled by the user.

The usage may look like this (assuming the user uses the convention of labelling apps with the `app` label):

```
kubectl apply --purge-missing-where 'app=myapp' -f https://example.com/myapp.json
```
