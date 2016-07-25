# Managed Namespaces

## Motivation

Currently `kubectl apply` will create any new resources that it sees, but it has no way of deleting anything.

There may be a need for some kind of `ApplySet` or `NamedList`, or other kind of a new top-level resource,
however it may be a good idea to avoid introducing anything new and see how we can use existing concepts.

Primary driver for this is simplification of how cluster add-ons are managed.

Most add-ons reside in `kube-system` namespace, it's thereby convenient to assume that this feature can be
implemented around namespaces, and no new abstraction is required for this use-case.

## Goal

Introduce optional `--purge-from-namespace` flag to `kubectl apply`, which will allow user to have resources
purged from the given namespace, if they haven't been explicitly declared as arguments to the same `kubectl apply`
invocation. This flag will not be compatible with `--namespace` and all resource must be in the same namespace,
set explicitly in each of the resource definitions. Alternative behaviour may be discussed, but the intention
is to avoid bad user experience.

There may exist a well-known annotation that can be used to exclude some resource(s) from being purged.

## Managing Add-ons

Let's first consider the add-ons use case.

There exists a `/etc/kubernetes/addons` directory with some number of files. All resources declared in those
files belong to the same namespace (`kube-system`), which is set explicitly in each of the resources.

When `kubectl apply --purge-from-namespace -f /etc/kubernetes/addons` runs for the first time all the resources will be created.

Eventually some new files appear in `/etc/kubernetes/addons` and some are modified.

Subsequent invocations of `kubectl apply --purge-from-namespace -f /etc/kubernetes/addons` result in new resources being created and
modified resources being updated.

When user deletes some of the files from `/etc/kubernetes/addons`, `kubectl apply --purge-from-namespace` would ensure those
files are deleted, based on the assumption that `/etc/kubernetes/addons` represents all resources that
are supposed to be present in `kube-system` namespace.


## Managing Any Other Application

It should be possible to apply the same approach to any other applications, with the assumption that single
application occupies a namespace of its own.

The usage may look like this:

```
kubectl apply --purge-from-namespaces -f https://example.com/myapp1.json -f https://example.com/myapp2.json
```

Given `myapp1.json` and `myapp2.json` both set their namespaces explicitly and those are not the same, their
resource would be purged, if no longer defined. In this case two invocations of `kubectl apply` for each of
the URLs will yield exactly the same results.

If `myapp1.json` and `myapp2.json` are in the same namespaces and it is explicitly defined, removing either
of the URLs from the invocations arguments will result in all the resources that URL has defined to be purged.

## Flag Dicscovery User Experience

As stated above, the `--purge-from-namespaces` flag shouldn't be enabled by default, however it should be discoverable.

Let's consider the following user experience.

User creates a new app with a deployment and a service.

```
app
├── deployment.yaml
└── service.yaml

0 directories, 2 files
```

They run `kubectl apply -f app` and deployment and service they have defined are being created in default namespaces.

Next, they run add `app/secret.yaml` and update `{deployment,service}.yaml`, then run `kubectl apply -f app`
again.

Now they decide to add `{deployment,service}2.yaml`, run `kubectl apply -f app`, but soon they realise that it would
be more appropriate to make this second service run in the same pod, so they refactor `deployment.yaml` and get rid of
`deployment2.yaml` and `service2.yaml`. Subsequent invocation of `kubectl apply -f app` prints this:

```
Based on revision history, it appears that you have resources that you may not wish to have any more.
By default, kubectl apply won't delete any resources, to see if you have any of such resources use `--purge-from-namespaces`.
```

> This message would only be appropriate to print in an interactive shell, and thereby it can be assumed that using
> either revision controll system or local command history would be safe.
