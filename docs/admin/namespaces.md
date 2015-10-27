<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Namespaces

## Abstract

A Namespace is a mechanism to partition resources created by users into
a logically named group.

## Motivation

A single cluster should be able to satisfy the needs of multiple users or groups of users (henceforth a 'user community').

Each user community wants to be able to work in isolation from other communities.

Each user community has its own:

1. resources (pods, services, replication controllers, etc.)
2. policies (who can or cannot perform actions in their community)
3. constraints (this community is allowed this much quota, etc.)

A cluster operator may create a Namespace for each unique user community.

The Namespace provides a unique scope for:

1. named resources (to avoid basic naming collisions)
2. delegated management authority to trusted users
3. ability to limit community resource consumption

## Use cases

1.  As a cluster operator, I want to support multiple user communities on a single cluster.
2.  As a cluster operator, I want to delegate authority to partitions of the cluster to trusted users
    in those communities.
3.  As a cluster operator, I want to limit the amount of resources each community can consume in order
    to limit the impact to other communities using the cluster.
4.  As a cluster user, I want to interact with resources that are pertinent to my user community in
    isolation of what other user communities are doing on the cluster.


## Usage

Look [here](namespaces/) for an in depth example of namespaces.

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

You can also get the summary of a specific namespace using:

```console
$ kubectl get namespaces <name>
```

Or you can get detailed information with:

```console
$ kubectl describe namespaces <name>
Name:	   default
Labels:	   <none>
Status:	   Active

No resource quota.

Resource Limits
 Type		Resource	Min	Max	Default
 ----				--------	---	---	---
 Container			cpu			-	-	100m
```

Note that these details show both resource quota (if present) as well as resource limit ranges.

Resource quota tracks aggregate usage of resources in the *Namespace* and allows cluster operators
to define *Hard* resource usage limits that a *Namespace* may consume.

A limit range defines min/max constraints on the amount of resources a single entity can consume in
a *Namespace*.

See [Admission control: Limit Range](../design/admission_control_limit_range.md)

A namespace can be in one of two phases:
   * `Active` the namespace is in use
   * `Terminating` the namespace is being deleted, and can not be used for new objects

See the [design doc](../design/namespaces.md#phases) for more details.

### Creating a new namespace

To create a new namespace, first create a new YAML file called `my-namespace.yaml` with the contents:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: <insert-namespace-name-here>
```

Note that the name of your namespace must be a DNS compatible label.

More information on the `finalizers` field can be found in the namespace [design doc](../design/namespaces.md#finalizers).

Then run:

```console
$ kubectl create -f ./my-namespace.yaml
```

### Working in namespaces

See [Setting the namespace for a request](../../docs/user-guide/namespaces.md#setting-the-namespace-for-a-request)
and [Setting the namespace preference](../../docs/user-guide/namespaces.md#setting-the-namespace-preference).

### Deleting a namespace

You can delete a namespace with

```console
$ kubectl delete namespaces <insert-some-namespace-name>
```

**WARNING, this deletes _everything_ under the namespace!**

This delete is asynchronous, so for a time you will see the namespace in the `Terminating` state.

## Namespaces and DNS

When you create a [Service](../../docs/user-guide/services.md), it creates a corresponding [DNS entry](dns.md).
This entry is of the form `<service-name>.<namespace-name>.svc.cluster.local`, which means
that if a container just uses `<service-name>` it will resolve to the service which
is local to a namespace.  This is useful for using the same configuration across
multiple namespaces such as Development, Staging and Production.  If you want to reach
across namespaces, you need to use the fully qualified domain name (FQDN).

## Design

Details of the design of namespaces in Kubernetes, including a [detailed example](../design/namespaces.md#example-openshift-origin-managing-a-kubernetes-namespace)
can be found in the [namespaces design doc](../design/namespaces.md)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/namespaces.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
