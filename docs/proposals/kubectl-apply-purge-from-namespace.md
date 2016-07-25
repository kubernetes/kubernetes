# Enable purging resources in `kubectl apply`

## Motivation

See [#19805](https://github.com/kubernetes/kubernetes/issues/19805).

Currently `kubectl apply` will create any new resources that exist in the resource file(s) it is supplied with, but it has no way of deleting anything that has been removed from such files.

This makes it difficult to use `kubectl apply` to update reality to match the declarative definitions of resources when that definition evolves over time, since changes can include deletions.

The primary driver for this use-case is to be able to manage cluster add-ons using `kubectl apply`.

## Proposed Solution

In a "simplest possible idea first" approach, this proposal suggests that we reuse namespaces as a de-facto way of grouping resources for the purpose of _purging_ (removing) resources which no longer exist.

Most add-ons reside in `kube-system` namespace, therefore for add-ons, an implementation based on namespaces would be reasonable.
The benefit of this is that no new abstraction (such as a new way of grouping resources) is required for this use-case.

Therefore, the proposal is to introduce an optional `--purge-from-namespace` flag to `kubectl apply`, which will allow a user to automatically purge resources from the given namespace, if they haven't been explicitly declared in files that are arguments to the same `kubectl apply` invocation.

All resources referred to must be explicitly labelled with a non-default namespace in order for the `--purge-from-namespace` argument to succeed.

This flag will require `--namespace` not to be set.
All referenced resources must be explicitly defined as being in the same namespace, set in each of the resource definitions.

The goal here is to avoid a negative user experience where the user accidentally deletes all the resources in some other namespace just because they accidentally refer to (e.g.) a single resource in some other namespace.

## Managing Add-ons

Let's first consider the add-ons use case.

There exists a `/etc/kubernetes/addons` directory with some number of files. All resources declared in those files belong to the same namespace (`kube-system`), which is set explicitly in each of the resources.

When `kubectl apply --purge-from-namespace -f /etc/kubernetes/addons` runs for the first time all the resources will be created.

Eventually some new files appear in `/etc/kubernetes/addons` and some are modified.

Subsequent invocations of `kubectl apply --purge-from-namespace -f /etc/kubernetes/addons` result in new resources being created and modified resources being updated.

When user deletes some of the files or resources within files from `/etc/kubernetes/addons`, `kubectl apply --purge-from-namespace` would ensure the resources that were referenced in the deleted files are deleted, based on the assumption that `/etc/kubernetes/addons` represents all resources that are supposed to be present in `kube-system` namespace.


## Managing Any Other Application

It should be possible to apply the same approach to any other applications, with the assumption that single application occupies a namespace of its own.

The usage may look like this:

```
kubectl apply --purge-from-namespace -f https://example.com/myapp1.json
```

Given resources in `myapp1.json` set their namespaces explicitly, resource that are no longer in this file would be purged, if no longer defined.

```
kubectl apply --purge-from-namespace -f https://example.com/myapp-part1.json -f https://example.com/myapp-part2.json
```

If `myapp1.json` and `myapp2.json` are in the same namespaces and it is explicitly defined, removing either of the URLs from the invocations arguments will result in all the resources that URL has defined to be purged.

## Flag Discovery User Experience

As stated above, the `--purge-from-namespace` flag shouldn't be enabled by default, however it should be discoverable.

So, if a `kubectl apply` command is run without `--purge-from-namespace`, and the referenced files are all in the same namespace, then the command could output something like:

```
$ kubectl apply -f resources.yaml
Updating ServiceA...
Updating ServiceB...
Updating DeploymentA...
The following resources in namespace `foo` are not referenced in the files you provided:

    DeploymentB
    ServiceC

You can automatically delete them, if you wish, using:

    kubectl apply --purge-from-namespace -f resources.yaml

$ kubectl apply --purge-from-namespace -f resources.yaml
Updating ServiceA...
Updating ServiceB...
Updating DeploymentA...
Deleting DeploymentB...
Deleting ServiceC...
Done!
```

If there are > 5 resources which match, we could just list the first 5.

If no namespace, or conflicting namespaces are provided in the resource file(s), this output would not be shown.

# Alternatives Considered

A new Kubernetes API-level concept "Versioned ApplySets" could record all the `kubectl apply` commands that were ever executed against a cluster.
This would allow deletions to be automatically detected and applied between invocations of `kubectl apply` without the onus of having to put resources into namespaces.

This adds complexity, since a new Kubernetes API object type would need to be invented.

It's not obvious how to make this concept work correctly in the case where one user runs `kubectl apply X` and later `kubectl apply X'` and a different user runs `kubectl apply Y` where `X'` is intended as an update to `X`
