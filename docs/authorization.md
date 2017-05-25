# Authorization
_This document describes features which are not yet implemented yet._

All API calls sent to the kubernetes API server are subject to authorization.

The authorization check for an API call compares attributes of the context of the API call, (such as user, resource type, and resource name) with a access policies.
An API call must be allowed by the policies in order to proceed.

## Requirements

The authorization mechanism should:
  - be expressive enough to represent a wide range of polices,
  - require little configuration for organizations that do not require complex policy
  - allow for complex policies
  - be able to express policy about object types created by kubernetes plugins.
  - allow for policy to be centrally stored and audited
  - allow for efficient evaluation
  - avoid complex precedence rules and complex policy expressions, both of which can make policy difficult to reason about.

### Use cases

These basic use cases should be possible:
 1. Restrict which users can add and remove machines to a cluster.
 1. Restrict which users can create or remove namespaces.
 1. Allow some users (e.g. cluster administrators) to create or remove any object in the cluster.
 1. Allow some users (e.g. project admins) to create or remove any object in the namespace
 1. Allow some users (e.g. project admins) to add or edit any policies scoped to the namespace
 1. Allow some users to create and delete pods and replicationControllers and services in a namespace, but not affect minions or modify or remove existing policies.
 1. Allow some users (e.g. developers) to create pods in namespace with a "dev" label but not a "prod" label.
 1. Allow some users to restart containers of a pod but not delete or modify it.

These additional use cases should be possible, but may require clients to write additional code or use a client binary:
 1. Allow a user to start a pod with label "prod" only if all the container images come from a certain repository and that repository has applied some metadata which indicates it was built by a trusted build process.
 1. Allow a cron controller to create a pod based on a certain podTemplate, but not create any other pod (see https://github.com/GoogleCloudPlatform/kubernetes/issues/170).
 1. Allow some users to create services in one IP range, but not in another one.
 1. Add a new user A with all the same policies as another user B.  (e.g. A and B are in the same LDAP group)
 1. View all policies and apply programatic sanity checks on them.
 1. Automatically expire policies some time after they were created.

### Prerequisites
This document assumes that [Namespaces](https://github.com/GoogleCloudPlatform/kubernetes/pull/1114) are implemented.  Access policies are often, but not always, attached to a namespace.  Namespaces have policies which control the ability of a user to access the namespace.

This document assumes that Kubernetes has support for _plugins_, which are processes separate from the apiserver, which can register new object types (e.g. a Build Controller), and to which the APIserver can delegate handling of those objects URLs (e.g. `/api/v1beta3/buildController`).  It is further assumed that there will be support libraries which can be used by plugins to efficiently watch and index objects and evaluate selectors.

This document assumes that Kubernetes has support for authenticating users via an identity provider.  That identity provider might be external to kubernetes.  The main mechanism for communicating an identity to the k8s apiserver from an identity provider is expected to be Oauth tokens used in an OpenID-like manner.  The authorization tokens may have scopes which restrict what k8s can learn about the party presenting the token, but, for the purposes of this document, they do not include scopes which refer to kubernetes objects or actions.  Whether and how to handle Oauth tokens which are scoped in terms of Kubernetes objects/action is future work.

This document assumes that Name is a user-provided typically-human-readable string which all Kubernetes API objects have, per #1124.

## Design Overview
Authorization occurs when a set of relevant _attributes_ are extracted from the current HTTP request, and passed to an `Authorize` function which allows or denies the action.

### Pluggability
Kubernetes will allow for multiple implementations of authorization.  The pattern for implementing authorization is:
  1. Implement the `Authorizer` interface in go code.  This can range from a trivial "always say yes" implementation, to one which makes RPC calls to an enterprise, third-party, or hosted authorization service.
  1. Optionally, write a plugin to add new Kinds of REST resource(s) to kubernetes API to store authorization policies.

The `GoogleCloudPlatform/kubernetes` source will include a default implementation which follows that pattern.

The APIserver calls the Authorizer interface:
```go
type Authorizer interface {
  Authorize(a Attributes)
}
```
to determine whether or not to allow each API action.
The APIserver will have a caching layer to avoid repeated calls to `Authorize` with the same arguments in a short time span.  As a consequence, policy changes will not take instant effect.

_TBD: It is not yet determined whether the pluing implementation will use redirects, will use the APIserver as a multiplexer.  In the latter case, the APIserver could handle authentication and authorization before muxing to plugins.  In the former case, plugins need to also implement authentication and authorization, perhaps with a common library, if they are also written in Go._

### Attributes

The APIserver composes a set of Attributes from the request to pass to the `Authorize` function.
```go
type Attributes struct {
  // Subject attributes
  User string       // a user identifier (e.g. "alice@example.com" or "alice.openid.example.org")
  Groups []string   // a list of groups the user belongs to, also from the IdProvider.

  // Verb
  Verb string

  // Object attributes
  Namespace string   // the namespace of the resource being accessed (empty if the resource is a namespace or object is not namespaced)
  Kind string        // the Kind of object
  Name string        // the Name of the object, per #1124 meaning of Name.
}
```

#### Subject attributes
The format and semantics of the User and Groups strings depends on the authentication setup.

*TBD: how groups list gets populated.  Does this come from identity provider, or from a kubernetes API resource, or somewhere else.  Getting it from the authentication provider is attractive when integrating with a system like LDAP that knows users and groups, or a system like OpenAM that stores membership data.  But it is not clear that group information is readily available from all identity providers.*


#### Verbs

A verb is a string which describes the action being taken.  Kubernetes core code defines the following mappings from (HTTP method, object Kind) to verb.

HTTP Method        | Resource prefix         | Verb
------------------ | ----------------------- | ----
GET                | /pods                   | core.read
POST,DELETE        | /pods                   | core.write
GET                | /services               | core.read
POST,DELETE        | /services               | core.write
GET                | /endpoints              | core.read
POST,DELETE        | /endpoints              | core.write
GET                | /replicationControllers | core.read
POST,DELETE        | /replicationControllers | core.write
GET                | /minions                | cluster.read
POST,DELETE        | /minions                | cluster.write
GET                | /bindings               | cluster.read
POST,DELETE        | /bindings               | cluster.write
GET                | /namespaces             | namespaces.read
POST,DELETE        | /namespaces             | namespaces.write

Plug-ins are required to define their own mappings.  Policies can use prefix matching on verbs, so that `core.*` includes `core.read` and `core.write`.
Several related Kinds of objects are often combined under a single Verb prefix to allow for expressing a policy with fewer policy objects (a user can be granted read access to pods, services, endpoints, and replicationControllers with a single verb, rather than four.)  When finer granularity is needed, the Kind object attribute can be used.  The reason for not doing away with prefixes like `core.` entirely is twofold:
   1. when a new plugin is introduced, users should not automatically get access to those new kinds of objects.
   1. read and write may not be the right verbs for some kinds of API actions.

*TBD: There are some actions which we plan to implement for kubernetes which do not neatly fit the GET/POST resource model, such as restarting a container, or creating a pod from a template.  These may be implemented with different URL paths, or with query parameters, or with flags in the payload.  However, it is done, there will need to be some mapping from the request to a verb.*

Some made-up but illustrative examples:

HTTP Method | request                 | Verb
----------- | ----------------------- | ----
POST        | /restartPod             | pods.restart
POST        | /pods/myPod?restart=1   | pods.restart
POST        | /pods/otherPod?fromtpl=myPodTpl | pods.write.fromtemplate

*Alternative: instead of using prefix matching, we could instead predefine a forest of verb precedences.  In role-based access control schemes like GCE, this is referred to as _concentric_ permissions: write includes read and so on.  The prefix matching format has the advantage that users don't have to memorize the precedence of verbs; this is not a problem in RBAC where there are only 3 or 4 verb, but in this scheme there may be many more and new ones may be introduced by plugins.*

#### Object attributes
The object of an API action is the kind and name of the object being accessed.

### HTTP responses

The HTTP responses for authentication and authorization failures are chosen to balance between allowing debugging and preventing information leakage.
 - If the user cannot be authenticated, then `401 Unauthorized` is returned, and `WWW-Authenticate` header is set.
    - Prevents leakage of information contained in namespace names.
    - `401 Unauthorized` and `WWW-Authenticate` required by some standards.
    - despite its name, `Unauthorized` appears to be the correct error code for authentication failures.
 - If the request is to GET an object that does not exist, return `404 Not Found`.
    - it was not found.
 - If the _subject_ attribute fails to matches the subject section of any policy with the empty namespace or the same namespace as the resource being requested, then return `404 Not Found`.
    - Prevents leakage of names of namespaces and names which the user does not have any association with, and which might contain sensitive information in a multi-tenant setup.
 - Last, check if the _subject_ is authorized to _verb_ that _object_.  If not, return `403 Forbidden`.
    - User can debug whether a call failed due to policy problem or incorrect name.

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
The `Authorize` function of the `Authorizer` interface compares the attributes passed to it with all the Policy objects.
Requests that do not match at least one Policy are not allowed.

Policy objects with an empty namespace have global effect.  Those with a non-empty namespace are only considered for objects in that same namespace.

Empty string Policy variable are treated and unset and so cannot match the corresponding attribute.

A string containing just "*" is considered to match all attributes.

Here is an reference implmentation of Policy matching code:
```go
func subjectMatch(p Policy, a Attributes) bool {
  if p.s.user != "" && p.s.user == a.s.user {
    // group is unset if user is set.  Validation code for Policy objects should ensure this.
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
  // p.v may have one star as the last character.  Validation ensures this.
  if p.v[:len(p.v)-1] == s.verb[:len(p.v)-1] { return 1 }
  return 0
}

func objectMatch(p Policy, a Attributes) bool {
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
  // empty namespace policy is global: always applies and not overrideable.
  // a policy with non-empty namespace only applies if the object has same namespace.
  if NamespaceOf(p) != "" && NamespaceOf(p) != a.o.Namespace {
    return 0
  }
  if subjectMatch(p, a) && verbMatch(p,a) && objectMatch(p, a) {
    return 1
  }
  return 0
}
```

## Examples

#### Restrict which users can add and remove machines to a cluster.
```
Policy{Effect{"ALLOW"}, Subject{Group{"cluster-admins"}}, Verb{"cluster.*"}} (Namespace of Policy = "")
```

#### Restrict which users can create or remove namespaces.
```
Policy{Effect{"ALLOW"}, Subject{Group{"ns-admins"}}, Verb{"namespaces.*"}} (Namespace of Policy = "")
```

#### Allow some users to create or remove any object in the cluster.

```
Policy{Effect{"ALLOW"}, Subject{Group{"cluster-admins"}}, Verb{"*"}}  (Namespace of Policy = "")
```

#### Allow some users to create or remove any object in the namespace
```
Policy{Effect{"ALLOW"}, Subject{Group{"some-ns-admins"}}, Verb{"*"},  (Namespace of Policy = "someNs")
```

Note: cluster resources are not in a namespace so only empty-namespace policies can ALLOW access to them.

#### Allow some users to add or edit any policies scoped to that namespace

```
Policy{Effect{"ALLOW"}, Subject{Group{"some-ns-managers"}}, Verb{"policies.*"} (Namespace of Policy = "someNs")}
```

#### Allow some users to view any policies scoped to that namespace

```
Policy{Effect{"ALLOW"}, Subject{Group{"some-ns-auditors"}}, Verb{"policy.read"}} (Namespace of Policy = "someNs")
```

#### Allow Bob, who writes buggy code, to create pods with a "stage=dev" label but not a "stage=prod" label.

```
Policy{Effect{"ALLOW"}, Subject{User{"bob"}}, Verb{"core.write"}, Object{Kind{"pods"}, Where{"stage=dev"} (Namespace of Policy = "someNs")
```

#### Allow some users to restart containers of a pod but not delete or modify it.
*TODO: need a verb for this.*

#### Allow a cron controller to create a pod based only on a certain podTemplate
*TODO: need to define a verb for this.*

## Policy Languages

As we gain experience with Policy objects, we may want to have a more concise representation.  This is outside the scope of the APIserver, and should be implemented by clients.  We may also want to be able to generate policy from templates, and to include it in config files.  Design of this may be better deferred until we have more experience with config.

