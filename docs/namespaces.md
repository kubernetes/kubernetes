# Kubernetes Proposal - Namespaces

**Related PR:** 

| Topic | Link |
| ---- | ---- |
| Identifiers.md | https://github.com/GoogleCloudPlatform/kubernetes/pull/1216 |
| Access.md | https://github.com/GoogleCloudPlatform/kubernetes/pull/891 |
| Indexing | https://github.com/GoogleCloudPlatform/kubernetes/pull/1183 |
| Cluster Subdivision | https://github.com/GoogleCloudPlatform/kubernetes/issues/442 |

## Background

High level goals:

* Enable an easy-to-use mechanism to logically scope Kubernetes resources
* Ensure extension resources to Kubernetes can share the same logical scope as core Kubernetes resources
* Ensure it aligns with access control proposal
* Ensure system has log n scale with increasing numbers of scopes

## Use cases

Actors:

1. k8s admin - administers a kubernetes cluster
2. k8s service - k8s daemon operates on behalf of another user (i.e. controller-manager)
2. k8s policy manager - enforces policies imposed on k8s cluster
3. k8s user - uses a kubernetes cluster to schedule pods

User stories:

1. Ability to set immutable namespace to k8s resources
2. Ability to list k8s resource scoped to a namespace
3. Restrict a namespace identifier to a DNS-compatible string to support compound naming conventions
4. Ability for a k8s policy manager to enforce a k8s user's access to a set of namespaces
5. Ability to set/unset a default namespace for use by kubecfg client
6. Ability for a k8s service to monitor resource changes across namespaces
7. Ability for a k8s service to list resources across namespaces

## Proposed Design

### Model Changes

Introduce a new attribute *Namespace* for each resource that must be scoped in a Kubernetes cluster.

A *Namespace* is a DNS compatible subdomain.

```
// TypeMeta is shared by all objects sent to, or returned from the client
type TypeMeta struct {
  Kind              string    `json:"kind,omitempty" yaml:"kind,omitempty"`
  Uid               string    `json:"uid,omitempty" yaml:"uid,omitempty"`
  CreationTimestamp util.Time `json:"creationTimestamp,omitempty" yaml:"creationTimestamp,omitempty"`
  SelfLink          string    `json:"selfLink,omitempty" yaml:"selfLink,omitempty"`
  ResourceVersion   uint64    `json:"resourceVersion,omitempty" yaml:"resourceVersion,omitempty"`
  APIVersion        string    `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
  Namespace         string    `json:"namespace,omitempty" yaml:"namespace,omitempty"`
  Name              string    `json:"name,omitempty" yaml:"name,omitempty"` 
}
```

An identifier, *UID*, is unique across time and space intended to distinguish between historical occurences of similar entities.

A *Name* is unique within a given *Namespace* at a particular time, used in resource URLs; provided by clients at creation time
and encouraged to be human friendly; intended to facilitate creation idempotence and space-uniqueness of singleton objects, distinguish
distinct entities, and reference particular entities across operations.

As of this writing, the following resources MUST have a *Namespace* and *Name*

* pod
* service
* replicationController
* endpoint

A *policy* MAY be associated with a *Namespace*.

If a *policy* has an associated *Namespace*, the resource paths it enforces are scoped to a particular *Namespace*.

## k8s API server

In support of namespace isolation, the Kubernetes API server will address resources by the following conventions:

The typical actors for the following requests are the k8s user or the k8s service.

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/ns/{ns}/{resourceType}/ | Create instance of {resourceType} in namespace {ns} |
| GET | GET | /api/{version}/ns/{ns}/{resourceType}/{name} | Get instance of {resourceType} in namespace {ns} with {name} |
| UPDATE | PUT | /api/{version}/ns/{ns}/{resourceType}/{name} | Update instance of {resourceType} in namespace {ns} with {name} |
| DELETE | DELETE | /api/{version}/ns/{ns}/{resourceType}/{name} | Delete instance of {resourceType} in namespace {ns} with {name} |
| LIST | GET | /api/{version}/ns/{ns}/{resourceType} | List instances of {resourceType} in namespace {ns} |
| WATCH | GET | /api/{version}/watch/ns/{ns}/{resourceType} | Watch for changes to a {resourceType} in namespace {ns} |

The typical actor for the following requests are the k8s service or k8s admin as enforced by k8s Policy.

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| WATCH | GET | /api/{version}/watch/{resourceType} | Watch for changes to a {resourceType} across all namespaces |
| LIST | GET | /api/{version}/list/{resourceType} | List instances of {resourceType} across all namespaces |

The legacy API patterns for k8s are an alias to interacting with the *default* namespace as follows.

| Action | HTTP Verb | Path | Description |
| ---- | ---- | ---- | ---- |
| CREATE | POST | /api/{version}/{resourceType}/ | Create instance of {resourceType} in namespace *default* |
| GET | GET | /api/{version}/{resourceType}/{name} | Get instance of {resourceType} in namespace *default* |
| UPDATE | PUT | /api/{version}/{resourceType}/{name} | Update instance of {resourceType} in namespace *default* |
| DELETE | DELETE | /api/{version}/{resourceType}/{name} | Delete instance of {resourceType} in namespace *default* |

The k8s API server verifies the *Namespace* on resource creation matches the *{ns}* on the path.

The k8s API server will enable efficient mechanisms to filter model resources based on the *Namespace*.  This may require
the creation of an index on *Namespace* that could support query by namespace with optional label selectors.

The k8s API server will associate a resource with a *Namespace* if not populated by the end-user based on the *Namespace* context
of the incoming request.  If the *Namespace* of the resource being created, or updated does not match the *Namespace* on the request,
then the k8s API server will reject the request.

TODO: Update to discuss k8s api server proxy patterns

## k8s storage

A namespace provides a unique identifier space and therefore must be in the storage path of a resource.

In etcd, we want to continue to still support efficient WATCH across namespaces.

Resources that persist content in etcd will have storage paths as follows:

/registry/{resourceType}/{resource.Namespace}/{resource.Name} 

This enables k8s service to WATCH /registry/{resourceType} for changes across namespace of a particular {resourceType}.

Upon scheduling a pod to a particular host, the pod's namespace must be in the key path as follows:

/host/{host}/pod/{pod.Namespace}/{pod.Name}

## k8s Authorization service

This design assumes the existence of an authorization service that filters incoming requests to the k8s API Server in order
to enforce user authorization to a particular k8s resource.  It performs this action by associating the *subject* of a request
with a *policy* to an associated HTTP path and verb.  This design encodes the *namespace* in the resource path in order to enable
external policy servers to function by resource path alone.  If a request is made by an identity that is not allowed by 
policy to the resource, the request is terminated.  Otherwise, it is forwarded to the apiserver.

## k8s controller-manager

The controller-manager will provision pods in the same namespace as the associated replicationController.

## k8s Kubelet

There is no major change to the kubelet introduced by this proposal.

### kubecfg client

kubecfg supports following:

```
kubecfg [OPTIONS] ns {namespace}
```

To set a namespace to use across multiple operations:

```
$ kubecfg ns ns1
```

To view the current namespace:

```
$ kubecfg ns
Using namespace ns1
```

To reset to the default namespace:

```
$ kubecfg ns default
```

In addition, each kubecfg request may explicitly specify a namespace for the operation via the following OPTION

--ns

When loading resource files specified by the -c OPTION, the kubecfg client will ensure the namespace is set in the
message body to match the client specified default.

If no default namespace is applied, the client will assume the following default namespace:

* default

The kubecfg client would store default namespace information in the same manner it caches authentication information today
as a file on user's file system.

