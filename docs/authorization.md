# Authorization
All API calls sent to the kubernetes API server are subject to authorization.

The authorization check for an API call compares attributes of the context of the API call, (such as user, resource type, and resource name) with a access policies.
An API call must be allowed by the policies in order to proceed.

## Namespaces
This document assumes that [Namespaces](https://github.com/GoogleCloudPlatform/kubernetes/pull/1114) are implemented.
Access policies are often, but not always, attached to a namespace.  Namespaces themselves can have policies which control the ability of a user to access the namespace.

## Requirements

### Use cases

Essential use cases:
 1. Restrict which users can add and remove machines to a cluster.
 1. Restrict which users can create or remove namespaces.
 1. Allow a user (cluster adminstrator) to create or remove any object.
 1. Allow a user (project admin) to create or remove any object in the project
 1. Allow a user (project admin) to add or edit any policies scoped to that project
 1. Allow a user (developer) to create pods and replicationControllers.
 1. Allow a user (developer) to create pods and replicationControllers and services.
 1. Allow a user (developer) to create pods in a project with a "dev" label but not a "prod" label.
 1. Allow a user (ops) to create pods with "prod" label (e.g. ones that are endpoints of a service that serves production requests).
 1. List policies which are relevant to a specific object.
 1. List policies which are relevant to a specific user.
 1. Allow a user to restart containers of a pod without modifying the pod object.

Bonus use cases:
 1. Allow a user to start a pod with label "prod" only if all the container images come from a certain repository and that repository has applied some metadata which indicates it was built by a trusted build process.
 1. Allow a cron controller to trigger the run a pod based on a certain podTemplate only.
 1.

8.  Can we define a Policy Template as a global resource?  Can we associate a Policy Template(s) with a Project Template to make initial setup easier?
9.  A policy object should have timestamps.
Typical scenario:
I have a global administrator for my OpenShift deployment that needs to access each Project.
Each project is managed by a team where each member may have particular roles with varying rights.
System view:
We have a Policy object that denotes global admin access added to each Project.
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
Option 1:
For each unique role, we would need to have a Policy object, but in this case our Policy subject would need to use an OR in the label selector that enumerated each person or group.  This could get awkward to manage.  If a policy object has timestamps, harder to reconcile when a person was added or removed from the project from this model.
Option 2:
For each person or group, we have a dedicated Policy object.  Simpler to enumerate, more resources to compare potentially.  Easier to audit when a person was added or removed from data model.
Issues:
There is a note that "Policy object is immutable, and is statically populated by admin"
It sounds like there is a need for a policy template, but don't you need to edit a policy in order to modify subjects to grow access?

## Authorization Flow

A common case of authorization flow is as follows:
```go
  if not UserAuthenticated() {
    w.WriteHeader(Unauthorized401);
    w.Header().Set("WWW-Authenticate", ...)
    return
  }
  if !Exists(NamespaceOf(req)) {
    w.WriteHeader(NotFound404);
    return
  }
  if !CanAccess(NamespaceOf(req), AttributesOf(user), AttributesOf(req)) {
    w.WriteHeader(NotFound404);
    return
  }
  if !Exists(ResourceOf(req)) {
    w.WriteHeader(NotFound404);
    return
  }
  if !CanAccess(ResourceOf(req), AttributesOf(user), AttributesOf(req)) {
    w.WriteHeader(Forbidden403);
    return
  }
```


Design discussion:
  1.  Authentication in Kubernetes is intended to be configurable to use different methods, but a common method is expected to be presenting an Oauth token which allows
the APIserver to confirm the user identity with a known, but not-necessarily part-of-kubernetes, identity provider.  In this case, the token does not contain scopes which refer to kubernetes objects or operations.  Whether and how to handle Oauth tokens which are scoped in terms of Kubernetes objects is future work.
  2. Unauthenticated users cannot determine what namespaces exists.  Prevents leakage of information contained in namespace names.
  3. Authenticated users cannot determine what namespaces exist, besides the ones they have access to.  Prevents information leakage from potentially large space of namespace names.  Right behavior when namespaces correspond to different tenants in a hosted product.
  4. User can debug whether a call failed due to policy problem or incorrect name.  This may be fairly common since an object creation may fail, and objects may be numerous and names generated by programs. 
  5.  A user authorized to access a naemspace can determine what object names exist in the namespace, including those the user is not authorized to read.  This small drawback is outweighed by the ability to debug.

## Policies and Attributes

An API action is considered to have a _subject_, a _verb_, and sometimes an _object_.
Each of these have one or more attributes.  All these attributes are passed to the
`CanAccess()` method.

Kubernetes stores PolicyStatement objects.
PolicyStatements also have _subject_, a _verb_, and _object_ sections.

The `CanAccess()` method compares the attributes passed to it with all the PolicyStatement objects.  When any PolicyStatement matches an API action, that API action is authorized.

###  Subject attributes

Subject attributes are:
  1. a user identifier (e.g. alice@example.com)
  1. the authentication provider (e.g. example.com or facebook.com)
  1. a list of groups that the user belongs to (from the authentication provider).

### Verbs
A verb is a single attribute.  It is a string which describes the action being taken.  Kubernetes core code defines mappings from HTTP method and `apiserver.RESTstorage` to verb name.  Plug-ins can define their own verbs or also map to the same verbs.

Suggested mappings:

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
POST        | /minions                | cluster.admin
DELETE      | /minions                | cluster.admin
GET         | /bindings               | cluster.read
POST        | /bindings               | cluster.write
DELETE      | /bindings               | cluster.write
GET         | /namespaces             | namespaces.read
POST        | /namespaces             | namespaces.write
DELETE      | /namespaces             | namespaces.write

### Object attributes
The object of an API action is the resource path that is being accessed, such as: `/pods/myPod`.
The labels of an object may also be used to filter access to sets of objects.

## PolicyStatement Objects

```go
type PolicyStatement {
    JSONBase
    a PolicyType
    s Subject
    v Verb
    o Object
    expires string // RFC3339
}

type Effect string
const {
   ALLOW Effect = "ALLOW",
   // Later DENY, etc.
}

type Subject string // Serialized label selector
type Verb string
type Object {
  exact string  // any resource with exactly this path
  // OR
  prefix string // any resource with this path prefix (after removing "/api/<version>/")
  // OR
  where string // serialized label selector.
}
```

# Policy Evaluation
Requests that do not match at least one PolicyStatement are not allowed.

Policy objects can belong to a namespace or not have a specified namespace, which means they are global.

TODO: no-namespace vs namespaced policies.  Think through more.
TODO: define DENY, and other operations and their precedence.

TODO: example of policy to delegate pod creation from a podTemplate (see https://github.com/GoogleCloudPlatform/kubernetes/issues/170).


In a simple implementation, the Authorize() module:
  - runs in the APIserver
  - searches all policy objects for a match.
  - updates its cache  when new Policy is added.

In alternate implementations, it may:
  - have indexes to speed matching.  (Maybe this can share code with Label Queries.)
  - defer to a centralized auth server for the enterprise. 


# Policy Language

# Use Cases

The authorization mechanism should:
  - to be expressive enough to represent a wide range of polices,
  - require little configuration for groups that do not require complex policy
  - allow for fairly complex policies for groups that require it.
  - allow for policy to be centrally stored and audited
  - allow for efficient evaluation

Policy Object:
Ideally, lists of policy objects would have relatively concise and readable YAML forms, such as:
```
{name: bob_can_read_pods, a: ALLOW, s: user.name is bob@example.com, v: GET, prefix: /pods}
{name: admins_can_delete_pods, a: ALLOW, s: user.role is admin, v: DELETE, prefix: /pods}
{name: tmp1234, a: ALLOW, s: user.name is "some.agent", v: POST, prefix: /pods/somepod, expires: 2014-08-13 16:21:42-07:00 }
```


policy languages vs policy objects.
VanillaST.

predefined labels in the selector or other policy language primitive for object type.
namespace for attributes.
namespace for other modules?

Policy objects are API objects.

 They express http://en.wikipedia.org/wiki/Attribute_Based_Access_Control 

Simple Profile:
- one Policy object that allows the single `userAccount` to CRUD objects in the single `project`.

Enterprise Profile:
- Many policy objects in each of many projects.
- Tools and services that wrap policy creation interface to enforce meta policies, do template expansions, report, etc.



Initial Features:
- Policy object is immutable
- Policy objects are statically populated in the K8s API store by reading a config file.  Only a K8s Cluster Admin can do this.
- Just a few policies per `project` which list which users can create objects, which can just view, them, etc.
- Objects are created with reference to these default policies.

Improvements:
- Have API calls to create and delete and modify Policy objects.   These would be in a separate API group from core K8s APIs.  This allows for replacing the K8s authorization service with an alternate implementation, and to centralize policies that might apply to services other than K8s.
- Ability to change policy for an object.
- Ability to create an object with a non-default policy effective immediately at creation time.
- Ability to defer policy object checking to a policy server.
- Authorization tokens to authorize entities without a `userAccount`.




### Blah
What is plan for verbs per resource type?
Seems like Client already defines these.

### AWS CloudFormation


wget https://raw.githubusercontent.com/marceldegraaf/blog-coreos-1/master/stack.yml
ruby -r json -r yaml -e "yaml = YAML.load(File.read('./stack.yml')); print yaml.to_json" > stack.json

### Concepts:
Enforcement point (reads a API call, adds attributes, and calls Decision Point)
Attribute Sources (adds attributes)
Decision Piont (evals policies) 

