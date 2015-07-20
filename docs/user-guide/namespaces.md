<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/namespaces.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Namespaces

Kubernetes supports multiple virtual clusters backed by the same physical cluster.
These virtual clusters are called namespaces.

## When to Use Multiple Namespaces

Namespaces are intended for use in environments with many users spread across multiple
teams, or projects.  For clusters with a few to tens of users, you should not
need to create or think about namespaces at all.  Start using namespaces when you
need the features they provide.

Namespaces provide a scope for names.  Names of resources need to be unique within a namespace, but not across namespaces.

Namespaces are a way to divide cluster resources between multiple uses (via [resource quota](../../docs/admin/resource-quota.md).

In future versions of Kubernetes, objects in the same namespace will have the same
access control policies by default.

It is not necessary to use multiple namespaces just to separate slightly different
resources, such as different versions of the same software: use [labels](#labels.md) to distinguish
resources within the same namespace.

## Working with Namespaces

Creation and deletion of namespaces is described in the [Admin Guide documentation
for namespaces](#../../docs/admin/namespaces.md)

### Viewing namespaces

You can list the current namespaces in a cluster using:

```console
$ kubectl get namespaces
NAME          LABELS    STATUS
default       <none>    Active
kube-system   <none>    Active
```

Kubernetes starts with two initial namespaces:
   * `default` The default namespace for objects with no other namespace
   * `kube-system` The namespace for objects created by the Kubernetes system

### Setting the namespace for a request

To temporarily set the namespace for a request, use the `--namespace` flag.

For example:

```console
$ kubectl --namespace=<insert-namespace-name-here> run nginx --image=nginx
$ kubectl --namespace=<insert-namespace-name-here> get pods
```

### Setting the namespace preference

You can permanently save the namespace for all subsequent kubectl commands in that
context.

First get your current context:

```console
$ export CONTEXT=$(kubectl config view | grep current-context | awk '{print $2}')
```

Then update the default namespace:

```console
$ kubectl config set-context $(CONTEXT) --namespace=<insert-namespace-name-here>
```

## Namespaces and DNS

When you create a [Service](services.md), it creates a corresponding [DNS entry](../admin/dns.md)1.
This entry is of the form `<service-name>.<namespace-name>.cluster.local`, which means
that if a container just uses `<service-name>` it will resolve to the service which
is local to a namespace.  This is useful for using the same configuration across
multiple namespaces such as Development, Staging and Production.  If you want to reach
across namespaces, you need to use the fully qualified domain name (FQDN).

## Not All Objects are in a Namespace

Most kubernetes resources (e.g. pods, services, replication controllers, and others) are
in a some namespace.  However namespace resources are not themselves in a namespace.
And, low-level resources, such as [nodes](../../docs/admin/node.md) and
persistentVolumes, are not in any namespace. Events are an exception: they may or may not
have a namespace, depending on the object the event is about.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/namespaces.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
