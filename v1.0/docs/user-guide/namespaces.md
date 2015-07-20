---
layout: docwithnav
---
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

{% highlight console %}
$ kubectl get namespaces
NAME          LABELS    STATUS
default       <none>    Active
kube-system   <none>    Active
{% endhighlight %}

Kubernetes starts with two initial namespaces:
   * `default` The default namespace for objects with no other namespace
   * `kube-system` The namespace for objects created by the Kubernetes system

You can also get the summary of a specific namespace using:

{% highlight console %}
$ kubectl get namespaces <name>
{% endhighlight %}

Or you can get detailed information with:

{% highlight console %}
$ kubectl describe namespaces <name>
Name:	   default
Labels:	   <none>
Status:	   Active

No resource quota.

Resource Limits
 Type		Resource	Min	Max	Default
 ----				--------	---	---	---
 Container			cpu			-	-	100m
{% endhighlight %}

Note that these details show both resource quota (if present) as well as resource limit ranges.

Resource quota tracks aggregate usage of resources in the *Namespace* and allows cluster operators
to define *Hard* resource usage limits that a *Namespace* may consume.

A limit range defines min/max constraints on the amount of resources a single entity can consume in
a *Namespace*.

See [Admission control: Limit Range](../design/admission_control_limit_range.md)

A namespace can be in one of two phases:
   * `Active` the namespace is in use
   * ```Terminating`` the namespace is being deleted, and can not be used for new objects

See the [design doc](../design/namespaces.md#phases) for more details.

### Creating a new namespace

To create a new namespace, first create a new YAML file called `my-namespace.yaml` with the contents:

{% highlight yaml %}
apiVersion: v1
kind: Namespace
metadata:
  name: <insert-namespace-name-here>
{% endhighlight %}

Note that the name of your namespace must be a DNS compatible label.

More information on the `finalizers` field can be found in the namespace [design doc](../design/namespaces.md#finalizers).

Then run:

{% highlight console %}
$ kubectl create -f ./my-namespace.yaml
{% endhighlight %}

### Setting the namespace for a request

To temporarily set the namespace for a request, use the `--namespace` flag.

For example:

{% highlight console %}
$ kubectl --namespace=<insert-namespace-name-here> run nginx --image=nginx
$ kubectl --namespace=<insert-namespace-name-here> get pods
{% endhighlight %}

### Setting the namespace preference

You can permanently save the namespace for all subsequent kubectl commands in that
context.

First get your current context:

{% highlight console %}
$ export CONTEXT=$(kubectl config view | grep current-context | awk '{print $2}')
{% endhighlight %}

Then update the default namespace:

{% highlight console %}
$ kubectl config set-context $(CONTEXT) --namespace=<insert-namespace-name-here>
{% endhighlight %}

### Deleting a namespace

You can delete a namespace with

{% highlight console %}
$ kubectl delete namespaces <insert-some-namespace-name>
{% endhighlight %}

**WARNING, this deletes _everything_ under the namespace!**

This delete is asynchronous, so for a time you will see the namespace in the `Terminating` state.

## Namespaces and DNS

When you create a [Service](services.md), it creates a corresponding [DNS entry](../admin/dns.md)1.
This entry is of the form `<service-name>.<namespace-name>.cluster.local`, which means
that if a container just uses `<service-name>` it will resolve to the service which
is local to a namespace.  This is useful for using the same configuration across
multiple namespaces such as Development, Staging and Production.  If you want to reach
across namespaces, you need to use the fully qualified domain name (FQDN).

### REST API

To interact with the Namespace API:

| Action | HTTP Verb | Path | Description |
| ------ | --------- | ---- | ----------- |
| CREATE | POST | /api/{version}/namespaces | Create a namespace |
| LIST | GET | /api/{version}/namespaces | List all namespaces |
| UPDATE | PUT | /api/{version}/namespaces/{namespace} | Update namespace {namespace} |
| DELETE | DELETE | /api/{version}/namespaces/{namespace} | Delete namespace {namespace} |
| FINALIZE | POST | /api/{version}/namespaces/{namespace}/finalize | Finalize namespace {namespace} |
| WATCH | GET | /api/{version}/watch/namespaces | Watch all namespaces |

To interact with content associated with a Namespace:

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/namespaces/{namespace}/{resourceType}/ | Create instance of {resourceType} in namespace {namespace} |
| GET | GET | /api/{version}/namespaces/{namespace}/{resourceType}/{name} | Get instance of {resourceType} in namespace {namespace} with {name} |
| UPDATE | PUT | /api/{version}/namespaces/{namespace}/{resourceType}/{name} | Update instance of {resourceType} in namespace {namespace} with {name} |
| DELETE | DELETE | /api/{version}/namespaces/{namespace}/{resourceType}/{name} | Delete instance of {resourceType} in namespace {namespace} with {name} |
| LIST | GET | /api/{version}/namespaces/{namespace}/{resourceType} | List instances of {resourceType} in namespace {namespace} |
| WATCH | GET | /api/{version}/watch/namespaces/{namespace}/{resourceType} | Watch for changes to a {resourceType} in namespace {namespace} |
| WATCH | GET | /api/{version}/watch/{resourceType} | Watch for changes to a {resourceType} across all namespaces |
| LIST | GET | /api/{version}/list/{resourceType} | List instances of {resourceType} across all namespaces |

## Design

Details of the design of namespaces in Kubernetes, including a [detailed example](../design/namespaces.md#example-openshift-origin-managing-a-kubernetes-namespace)
can be found in the [namespaces design doc](../design/namespaces.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/namespaces.html?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

