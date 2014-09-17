# Authorization
All API calls sent to the kubernetes API server are subject to authorization.

The authorization check for an API call compares attributes of the context of the API call, (such as user, resource type, and resource name) with a access policies.
An API call must be allowed by the policies in order to proceed.

## Requirements

The authorization mechanism should:
  - to be expressive enough to represent a wide range of polices,
  - require little configuration for organizations that do not require complex policy
  - allow for fairly complex policies for groups that require it.
  - be able to express policy about object types created by kubernetes plugins.
  - allow for policy to be centrally stored and audited
  - allow for efficient evaluation
  - avoid large numbers of precedence rules and avoid excessive expressiveness in policy matching operators, which can make policy difficult to reason about.

### Use cases

These basic use cases should be possible to accomplish using the kubernetes source or binary distribution as-is (once authentication is setup):
 1. Restrict which users can add and remove machines to a cluster.
 1. Restrict which users can create or remove namespaces.
 1. Allow some users (e.g. cluster administrators) to create or remove any object in the cluster.
 1. Allow some users (e.g. project admins) to create or remove any object in the project
 1. Allow some users (e.g. project admins) to add or edit any policies scoped to that project
 1. Allow some users to create and delete pods and replicationControllers and services in a project, but not affect minions or modify or remove existing policies.
 1. Allow some users (e.g. developers) to create pods in a project with a "dev" label but not a "prod" label.
 1. Allow some users to restart containers of a pod but not delete or modify it.

These additional use cases should be possible, but may require clients to write additional code or use a client binary:
 1. Allow a user to start a pod with label "prod" only if all the container images come from a certain repository and that repository has applied some metadata which indicates it was built by a trusted build process.
 1. Allow a cron controller to create a pod based on a certain podTemplate, but not create any other pod (see https://github.com/GoogleCloudPlatform/kubernetes/issues/170).
 1. Allow a user (e.g. auditor) to list policies which are relevant to a specific object.
 1. Allow a user (e.g. auditor) to list policies which are relevant to a specific user.
 1. Allow some users to create services in one IP range, but not in another one.
 1. Add a new user A with all the same policies as another user B.  (e.g. A and B are in the same LDAP group)
 1. View all policies and apply programatic sanity checks on them.
 1. Automatically expire policies some time after they were created.

### Prerequisites
This document assumes that [Namespaces](https://github.com/GoogleCloudPlatform/kubernetes/pull/1114) are implemented.  Access policies are often, but not always, attached to a namespace.  Namespaces have policies which control the ability of a user to access the namespace.

This document assumes that Kubernetes has support for authenticating users via an identity provider.  That identity provider might be external to kubernetes, or it might share a process with the apiserver.  In either case, its concepts and objects are not part of the kubernetes core.  The main mechanism for communicating an identity to the k8s apiserver from an identity provider is expected to be Oauth tokens used in an OpenID-like manner.  The authorization tokens may have scopes which restrict what k8s can learn about the party presenting the token, but, for the purposes of this document, they do not include scopes which refer to kubernetes objects or actions.  Whether and how to handle Oauth tokens which are scoped in terms of Kubernetes objects is future work.

This document assumes that Kubernetes has support for _plugins_, which are processes separate from the apiserver, which can register new object types (e.g. a Build Controller), and which the APIserver can delegate handling of those objects URLs (e.g. `/api/v1beta3/buildController`).  It is further assumed that there will be support libraries which can be used by plugins to efficiently watch and index objects and evaluate selectors.

This document assumes that Name is a user-provided typically-human-readable string, per #1124.

## Design Overview
Authorization occurs when a set of relevant _attributes_ is extracted from the current HTTP request, and passed to an `Authorize` function, which allows or denies the action.

### Pluggability
Kubernetes will allow for multiple implementations of authorization.  The pattern for implementing your own authorization is:
  1. Implement the `Authorizer` interface in go code.  This can range from a trivial "always say yes" implementation, to one which makes RPC calls to an enterprise or hosted authentication service.
  1. Optionally, write a plugin to add REST resource Kinds to kubernetes API to store authorization policies.

The `GoogleCloudPlatform/kubernetes` source will include a default implementation which follows that pattern.

The APIserver calls the Authorizer interface:
```go
type Authorizer interface {
  Authorize(a Attributes)
}
```
to determine whether or not to allow each API action.
The APIserver will have a caching layer to avoid calls to `Authorize` when there are several requests in a short time with the same attributes.

_TBD: depending on the final plugin design, an API plugin, which is a separate processes from the apiserver, might either handle API actions after the APIserver has done authorization , or do Authorization itself for object Kinds that it owns, using a common library._

### Attributes

The APIserver composes a set of Attributes from the request to pass to the `Authorize` function.
```go
type Attributes struct {
  // Subject attributes
  User string       // a user identifier (e.g. alice@example.com, "alice.openid.example.org")
  Groups []string   // a list of groups the user belongs to, also from the IdProvider.

  // Verb
  Verb string

  // Object attributes
  Namespace string   // the namespace of the resource being accessed (empty if the resource is a namespace or object is not namespaced)
  Kind string        // the Kind of object
  Name string        // the Name of the object, per #1124 meaning of Name.
  Resource string   // the REST url of the resource being accessed, e.g. "/pods/myPod", or empty if not a single object, as when listing.
}
```

#### Subject attributes
The meaning of the User string depends on the authentication setup, which may vary across kubernetes clusters but would be consistent for a given cluster.

*TBD: how groups list gets populated.  Does this come from identity provider, or from a kubernetes API resource, or somewhere else.*

#### Verbs
A verb is a string which describes the action being taken.  Kubernetes core code defines mappings from (HTTP method, `apiserver.RESTstorage`) to verb name.  Plug-ins can do the same.

HTTP Method | Resource prefix         | Verb
----------- | ----------------------- | ----
GET         | /pods                   | core.read
POST        | /pods                   | core.write
DELETE      | /pods                   | core.write
GET         | /services               | core.read
POST        | /services               | core.write
DELETE      | /services               | core.write
GET         | /endpoints              | core.read
POST        | /endpoints              | core.write
DELETE      | /endpoints              | core.write
GET         | /replicationControllers | core.read
POST        | /replicationControllers | core.write
DELETE      | /replicationControllers | core.write
GET         | /minions                | cluster.read
POST        | /minions                | cluster.write
DELETE      | /minions                | cluster.write
GET         | /bindings               | cluster.read
POST        | /bindings               | cluster.write
DELETE      | /bindings               | cluster.write
GET         | /namespaces             | namespaces.read
POST        | /namespaces             | namespaces.write
DELETE      | /namespaces             | namespaces.write

*TBD: is this the right set of verbs?  a balance is needed between having too few verbs, which prevents fine grain access control, and too many verbs, which results in too many policies.*

#### Object attributes
The object of an API action is the resource path that is being accessed, such as: `/pods/myPod`.

### HTTP responses

The HTTP responses for various conditions are chosen to balance between allowing debugging and preventing information leakage.
 - If the user cannot be authenticated, then `401 Unauthorized` is returned, and `WWW-Authenticate` header is set.  Reason:
    - Prevents leakage of information contained in namespace names.
    - `401 Unauthorized` and `WWW-Authenticate` required by some standards.
 - If the request is GET, then check if the user is authorized to `namespace.read` the namespace of the objects being read/listed.  If not, return `404 Not Found`.  Reason:
    - Prevents information leakage from potentially large space of namespace names, especially in a multi-tenant setup.
    - Lets user know that authentication succeeded.
 - If the request is POST or DELETE, then check if the user is authorized to `namespace.read` the namespace of the objects being read/listed.  If not, return `404 Not Found`.  Reason:
    - Prevents information leakage from potentially large space of namespace names, especially in a multi-tenant setup.
    - Lets user know that authentication succeeded.
 - If the request is to GET an object that does not exist, return `404 Not Found`.  Reason:
    - it was not found.
 -  If the request is to GET an object that exists, to list multiple objects, or to POST or DELETE; then check if the _subject_ is authorized to _verb_ that _object_.  If not, return `403 Forbidden`.  Reason:
    - User can debug whether a call failed due to policy problem or incorrect name.  This may be fairly common since an object creation may fail, and objects may be numerous and names generated by programs.

## Default Implementation

*The rest of this document refers to the design of the default implementation of authorization.*

This implementation will introduce a `Policy` kind of object, which can be created and deleted using the K8s API.  Access to `Policy` is in turn subject to authorization control.
It also implements the `Authorizer` go interface.

Each call to `Authorize` results in the passed-in attributed being matched compared against every relevant `Policy`.

## Policy Objects

```go
type Effect string
const {
   ALLOW Effect = "ALLOW",
   // TODO: add DENY, etc if required.
}

type User string
type Group string

type Subject {
  // One of the following:
  user User // a user identifier (e.g. alice@example.com) with meaning in the context of the identity provider.
  group Group // a group identifier (e.g. foo-project-admins) with meaning in the context of the identity provider.
}

type Verb string

type Kind string
type ExactName string
type Selector string

type Object {
  // Optional
  kind Kind
  // Zero or one of the following:
  exactName ExactName // this exact name
  where Selector // any object matchting this stringified label selector
}

type Policy {
    JSONBase
    a Effect
    s Subject
    v Verb
    o Object
    expires string // RFC3339
}
```

### Policy Evaluation
The `Authorizer.Authorize` method compares the attributes passed to it with all the Policy objects.
Requests that do not match at least one Policy are not allowed.

Policy objects can belong to a namespace or not have a specified namespace, which means they are global.

Zero-valued Policy variables are treated as unset.  A policy with everything unset matches nothing.  So, you cannot write a policy that allows everything.  If this behavior is needed, a flag can be used to replace the Policy plugin with an default allow-all authorizer.

A Policy matches the attributes if:
```go
func subjectMatch(p Policy, a Attributes) bool {
  if p.s.user != "" && p.s.user == a.s.user {
    // validation ensures group is unset
    return 1
  }
  if p.group != "" && stringInSlice(p.group, a.groups) {
    return 1
  }
  return 0
}

func verbMatch(p Policy, a Attributes) bool {
  // empty p.v matches nothing, because s.verb should always be non-empty
  if p.v == s.verb { return 1 }
  return 0
}

func objectMatch(p Policy, a Attributes) bool {
  // empty namespace policy is global: always applies and not overrideable.
  // a policy with non-empty namespace only applies if the object has same namespace.
  if NamespaceOf(p) != "" && NamespaceOf(p) != a.Namespace {
    return 0
  }
  if p.o.kind != "" && p.o.kind == a.Kind { return 1 }
  if p.o.exact_name != "" && p.o.name == a.Name { return 1 }
  if p.o.where != "" {
    matches = NamesOf(DoSelector(p.o.where, NamespaceOf(p))) {
    if stringInSlice(p.o.name, matches) {
      return 1
    }
  }
  return 0
}

func matches(p Policy, a Attributes) bool {
  if p.expires < now { return 0 }
  if subjectMatch(p, a) && verbMatch(p,a) && objectMatch(p, a) {
    return 1
  }
  return 0
}
```

## Examples

Here are policies that would implement each of the basic use cases listed above.

TODO: go through these, making sure read and write is given and namespace is given.

### Restrict which users can add and remove machines to a cluster.
One user:
```
Policy{Effect{"ALLOW"}, Subject{User{"bob"}}, Verb{"cluster.read"}}
Policy{Effect{"ALLOW"}, Subject{User{"bob"}}, Verb{"cluster.write"}}
```

A group "cluster-admins", defined elsewhere:
```
Policy{Effect{"ALLOW"}, Subject{Group{"cluster-admins"}}, Verb{"cluster.read"}}
Policy{Effect{"ALLOW"}, Subject{Group{"cluster-admins"}}, Verb{"cluster.write"}}
```
### Restrict which users can create or remove namespaces.
```
Policy{Effect{"ALLOW"}, Subject{Group{"ns-admins"}}, Verb{"namespaces.read"}}
Policy{Effect{"ALLOW"}, Subject{Group{"ns-admins"}}, Verb{"namespaces.write"}}
```

### Allow some users to create or remove any object in the cluster.

_TODO: this is not possible without enumerating all verbs, and that list might grow in the presence of plugins._
```
Policy{Effect{"ALLOW"}, Subject{Group{"cluster-junior-admins"}}, Verb{"core.write"}}
Policy{Effect{"ALLOW"}, Subject{Group{"cluster-junior-admins"}}, Verb{"cluster.write"}}
Policy{Effect{"ALLOW"}, Subject{Group{"cluster-junior-admins"}}, Verb{"namespaces.write"}}
Policy{Effect{"ALLOW"}, Subject{Group{"cluster-junior-admins"}}, Verb{"policy.write"}}
```
### Allow some users to create or remove any object in the project

_TODO: this is not possible without enumerating all verbs, and that list might grow in the presence of plugins._

```
Policy{Effect{"ALLOW"}, Subject{Group{"my-ns-admins"}}, Verb{"core.write"}, Object{Namespace{"someNs"}}
Policy{Effect{"ALLOW"}, Subject{Group{"my-ns-admins"}}, Verb{"namespaces.write"}, Object{Namespace{"someNs"}}
Policy{Effect{"ALLOW"}, Subject{Group{"my-ns-admins"}}, Verb{"policy.write"}, Object{Namespace{"someNs"}}
```

### Allow some users to add or edit any policies scoped to that project

```
Policy{Effect{"ALLOW"}, Subject{Group{"my-ns-managers"}}, Verb{"policy.write"}, Object{Namespace{"someNs"}}
```
### Allow some users to create and delete pods and replicationControllers and services in a project
### Allow some users (e.g. developers) to create pods in a project with a "dev" label but not a "prod" label.
### Allow some users to restart containers of a pod but not delete or modify it.

TODO: examples for advanced use cases listed above.

### Complex example.
Each project is managed by a team where each member may have particular roles with varying rights.
System view:
We have a Policy object that denotes global admin access added to each Project.
```
Policy{
    project: "the project", 
    name:"openshift_admin",
    prefix: "/", 
    s:"group=openshift_admin"
    v: "ANY"
    o: {
        prefix: "/"
    }
}
```
Option 1:
For each unique role, we would need to have a Policy object, but in this case our Policy subject would need to use an OR in the label selector that enumerated each person or group.  This could get awkward to manage.  If a policy object has timestamps, harder to reconcile when a person was added or removed from the project from this model.
Option 2:
For each person or group, we have a dedicated Policy object.  Simpler to enumerate, more resources to compare potentially.  Easier to audit when a person was added or removed from data model.
Issues:
There is a note that "Policy object is immutable, and is statically populated by admin"
It sounds like there is a need for a policy template, but do not you need to edit a policy in order to modify subjects to grow access?

## Policy Languages

As we gain experience with Policy objects, we may want to have a more concise representation.  This is outside the scope of the APIserver, and should be implemented by clients.  We may also want to be able to generate policy from templates, and to include it in config files.  Design of this may be better deferred until we have more experience with config.

