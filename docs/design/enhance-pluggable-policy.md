<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Enhance Pluggable Policy

While trying to develop an authorization plugin for Kubernetes, we found a few
places where API extensions would ease development and add power.  There are a
few goals:
 1.  Provide an authorization plugin that can evaluate a .Authorize() call based
on the full content of the request to RESTStorage.  This includes information
like the full verb, the content of creates and updates, and the names of
resources being acted upon.
 1.  Provide a way to ask whether a user is permitted to take an action without
 running in process with the API Authorizer.  For instance, a proxy for exec
 calls could ask whether a user can run the exec they are requesting.
 1.  Provide a way to ask who can perform a given action on a given resource.
This is useful for answering questions like, "who can create replication
controllers in my namespace".

This proposal adds to and extends the existing API to so that authorizers may
provide the functionality described above.  It does not attempt to describe how
the policies themselves can be expressed, that is up the authorization plugins
themselves.


## Enhancements to existing Authorization interfaces

The existing Authorization interfaces are described
[here](../admin/authorization.md). A couple additions will allow the development
of an Authorizer that matches based on different rules than the existing
implementation.

### Request Attributes

The existing authorizer.Attributes only has 5 attributes (user, groups,
isReadOnly, kind, and namespace). If we add more detailed verbs, content, and
resource names, then Authorizer plugins will have the same level of information
available to RESTStorage components in order to express more detailed policy.
The replacement excerpt is below.

An API request has the following attributes that can be considered for
authorization:
  - user - the user-string which a user was authenticated as. This is included
in the Context.
  - groups - the groups to which the user belongs. This is included in the
Context.
  - verb - string describing the requesting action. Today we have: get, list,
watch, create, update, and delete. The old `readOnly` behavior is equivalent to
allowing get, list, watch.
  - namespace - the namespace of the object being access, or the empty string if
the endpoint does not support namespaced objects.  This is included in the
Context.
  - resourceGroup - the API group of the resource being accessed
  - resourceVersion - the API version of the resource being accessed
  - resource - which resource is being accessed
    - applies only to the API endpoints, such as `/api/v1beta1/pods`. For
miscellaneous endpoints, like `/version`, the kind is the empty string.
  - resourceName - the name of the resource during a get, update, or delete
action.
  - subresource - which subresource is being accessed

A non-API request has 2 attributes:
  - verb - the HTTP verb of the request
  - path - the path of the URL being requested


### Authorizer Interface

The existing Authorizer interface is very simple, but there isn't a way to
provide details about allows, denies, or failures. The extended detail is useful
for UIs that want to describe why certain actions are allowed or disallowed. Not
all Authorizers will want to provide that information, but for those that do,
having that capability is useful. In addition, adding a `GetAllowedSubjects`
method that returns back the users and groups that can perform a particular
action makes it possible to answer questions like, "who can see resources in my
namespace" (see [ResourceAccessReview](#ResourceAccessReview) further down).

```go
// OLD
type Authorizer interface {
  Authorize(a Attributes) error 
}
```

```go
// NEW
// Authorizer provides the ability to determine if a particular user can perform
// a particular action
type Authorizer interface {
  // Authorize takes a Context (for namespace, user, and traceability) and
  //   Attributes to make a policy determination. 
  // reason is an optional return value that can describe why a policy decision
  //   was made.  Reasons are useful during debugging when trying to figure out
  //   why a user or group has access to perform a particular action.
  Authorize(ctx api.Context, a Attributes) (allowed bool, reason string, evaluationError error)
}

// AuthorizerIntrospection is an optional interface that provides the ability to
//   determine which users and groups can perform a particular action. This is
//   useful for building caches of who can see what. For instance, "which
//   namespaces can this user see".  That would allow someone to see only the
//   namespaces they are allowed to view instead of having to choose between
//   listing them all or listing none.
type AuthorizerIntrospection interface {
  // GetAllowedSubjects takes a Context (for namespace and traceability) and 
  // Attributes to determine which users and groups are allowed to perform the
  // described action in the namespace. This API enables the ResourceBasedReview
  // requests below
  GetAllowedSubjects(ctx api.Context, a Attributes) (users util.StringSet, groups util.StringSet, evaluationError error)
}
```

### SubjectAccessReviews

This set of APIs answers the question: can a user or group (use authenticated
user if none is specified) perform a given action. Given the Authorizer
interface (proposed or existing), this endpoint can be implemented generically
against any Authorizer by creating the correct Attributes and making an
.Authorize() call.

There are three different flavors:

1. `/apis/authorization.kubernetes.io/{version}/subjectAccessReviews` - this
checks to see if a specified user or group can perform a given action at the
cluster scope or across all namespaces. This is a highly privileged operation.
It allows a cluster-admin to inspect rights of any person across the entire
cluster and against cluster level resources.
2. `/apis/authorization.kubernetes.io/{version}/personalSubjectAccessReviews` -
this checks to see if the current user (including his groups) can perform a
given action at any specified scope. This is an unprivileged operation. It
doesn't expose any information that a user couldn't discover simply by trying an
endpoint themselves.
3. `/apis/authorization.kubernetes.io/{version}/ns/{namespace}/localSubjectAccessReviews` -
this checks to see if a specified user or group can perform a given action in
**this** namespace. This is a moderately privileged operation. In a multi-tenant
environment, having a namespace scoped resource makes it very easy to reason
about powers granted to a namespace admin. This allows a namespace admin
(someone able to manage permissions inside of one namespaces, but not all
namespaces), the power to inspect whether a given user or group can manipulate
resources in his namespace.

SubjectAccessReview is runtime.Object with associated RESTStorage that only
accepts creates. The caller POSTs a SubjectAccessReview to this URL and he gets
a SubjectAccessReviewResponse back. Here is an example of a call and its
corresponding return:

```
// input
{
  "kind": "SubjectAccessReview",
  "apiVersion": "authorization.kubernetes.io/v1",
  "authorizationAttributes": {
    "verb": "create",
    "resource": "pods",
    "user": "Clark",
    "groups": ["admins", "managers"]
  }
}

// POSTed like this
curl -X POST /apis/authorization.kubernetes.io/{version}/subjectAccessReviews -d @subject-access-review.json
// or 
accessReviewResult, err := Client.SubjectAccessReviews().Create(subjectAccessReviewObject)

// output
{
  "kind": "SubjectAccessReviewResponse",
  "apiVersion": "authorization.kubernetes.io/v1",
  "allowed": true
}
```

PersonalSubjectAccessReview is runtime.Object with associated RESTStorage that
only accepts creates. The caller POSTs a PersonalSubjectAccessReview to this URL
and he gets a SubjectAccessReviewResponse back. Here is an example of a call and
its corresponding return:

```
// input
{
  "kind": "PersonalSubjectAccessReview",
  "apiVersion": "authorization.kubernetes.io/v1",
  "authorizationAttributes": {
    "verb": "create",
    "resource": "pods",
    "namespace": "any-ns",
  }
}

// POSTed like this
curl -X POST /apis/authorization.kubernetes.io/{version}/personalSubjectAccessReviews -d @personal-subject-access-review.json
// or
accessReviewResult, err := Client.PersonalSubjectAccessReviews().Create(subjectAccessReviewObject)

// output
{
  "kind": "PersonalSubjectAccessReviewResponse",
  "apiVersion": "authorization.kubernetes.io/v1",
  "allowed": true
}
```

LocalSubjectAccessReview is runtime.Object with associated RESTStorage that only
accepts creates. The caller POSTs a LocalSubjectAccessReview to this URL and he
gets a LocalSubjectAccessReviewResponse back. Here is an example of a call and
its corresponding return:

```
// input
{
  "kind": "LocalSubjectAccessReview",
  "apiVersion": "authorization.kubernetes.io/v1",
  "namespace": "my-ns"
  "authorizationAttributes": {
    "verb": "create",
    "resource": "pods",
    "user": "Clark",
    "groups": ["admins", "managers"]
  }
}

// POSTed like this
curl -X POST /apis/authorization.kubernetes.io/{version}/localSubjectAccessReviews -d @local-subject-access-review.json
// or 
accessReviewResult, err := Client.LocalSubjectAccessReviews().Create(localSubjectAccessReviewObject)

// output
{
  "kind": "LocalSubjectAccessReviewResponse",
  "apiVersion": "authorization.kubernetes.io/v1",
  "namespace": "my-ns"
  "allowed": true
}
```

The actual Go objects look like this:

```go
type AuthorizationAttributes struct {
  // Namespace is the namespace of the action being requested. Currently, there
  // is no distinction between no namespace and all namespaces
  Namespace string `json:"namespace" description:"namespace of the action being requested"`
  // Verb is one of: get, list, watch, create, update, delete
  Verb string `json:"verb" description:"one of get, list, watch, create, update, delete"`
  // Resource is one of the existing resource types
  ResourceGroup string `json:"resourceGroup" description:"group of the resource being requested"`
  // ResourceVersion is the version of resource
  ResourceVersion string `json:"resourceVersion" description:"version of the resource being requested"`
  // Resource is one of the existing resource types
  Resource string `json:"resource" description:"one of the existing resource types"`
  // ResourceName is the name of the resource being requested for a "get" or
  // deleted for a "delete"
  ResourceName string `json:"resourceName" description:"name of the resource being requested for a get or delete"`
  // Subresource is one of the existing subresources types
  Subresource string `json:"subresource" description:"one of the existing subresources"`
}

// SubjectAccessReview is an object for requesting information about whether a
// user or group can perform an action
type SubjectAccessReview struct {
  kapi.TypeMeta `json:",inline"`

  // AuthorizationAttributes describes the action being tested.
  AuthorizationAttributes `json:"authorizationAttributes" description:"the action being tested"`
  // User is optional, but at least one of User or Groups must be specified
  User string `json:"user" description:"optional, user to check"`
  // Groups is optional, but at least one of User or Groups must be specified
  Groups []string `json:"groups" description:"optional, list of groups to which the user belongs"`
}

// SubjectAccessReviewResponse describes whether or not a user or group can
// perform an action
type SubjectAccessReviewResponse struct {
  kapi.TypeMeta

  // Allowed is required.  True if the action would be allowed, false otherwise.
  Allowed bool
  // Reason is optional.  It indicates why a request was allowed or denied.
  Reason string
}

// PersonalSubjectAccessReview is an object for requesting information about
// whether a user or group can perform an action
type PersonalSubjectAccessReview struct {
  kapi.TypeMeta `json:",inline"`

  // AuthorizationAttributes describes the action being tested.
  AuthorizationAttributes `json:"authorizationAttributes" description:"the action being tested"`
}

// PersonalSubjectAccessReviewResponse describes whether this user can perform
// an action
type PersonalSubjectAccessReviewResponse struct {
  kapi.TypeMeta

  // Namespace is the namespace used for the access review
  Namespace string
  // Allowed is required.  True if the action would be allowed, false otherwise.
  Allowed bool
  // Reason is optional.  It indicates why a request was allowed or denied.
  Reason string
}

// LocalSubjectAccessReview is an object for requesting information about
// whether a user or group can perform an action
type LocalSubjectAccessReview struct {
  kapi.TypeMeta `json:",inline"`

  // AuthorizationAttributes describes the action being tested.
  AuthorizationAttributes `json:"authorizationAttributes" description:"the action being tested"`
  // User is optional, but at least one of User or Groups must be specified
  User string `json:"user" description:"optional, user to check"`
  // Groups is optional, but at least one of User or Groups must be specified
  Groups []string `json:"groups" description:"optional, list of groups to which the user belongs"`
}

// LocalSubjectAccessReviewResponse describes whether or not a user or group can
// perform an action
type LocalSubjectAccessReviewResponse struct {
  kapi.TypeMeta

  // Namespace is the namespace used for the access review
  Namespace string
  // Allowed is required.  True if the action would be allowed, false otherwise.
  Allowed bool
  // Reason is optional.  It indicates why a request was allowed or denied.
  Reason string
}
```

### ResourceAccessReview

This set of APIs nswers the question: which users and groups can perform the
specified verb on the specified resourceKind. Given the Authorizer interface
described above, this endpoint can be implemented generically against any
Authorizer by calling the .GetAllowedSubjects() function.

There are two different flavors:

1. `/apis/authorization.kubernetes.io/{version}/resourceAccessReview` - this
checks to see which users and groups can perform a given action at the cluster
scope or across all namespaces. This is a highly privileged operation. It allows
a cluster-admin to inspect rights of all subjects across the entire cluster and
against cluster level resources.
2. `/apis/authorization.kubernetes.io/{version}/ns/{namespace}/localResourceAccessReviews` -
this checks to see which users and groups can perform a given action in **this**
namespace. This is a moderately privileged operation. In a multi-tenant
environment, having a namespace scoped resource makes it very easy to reason
about powers granted to a namespace admin. This allows a namespace admin
(someone able to manage permissions inside of one namespaces, but not all
namespaces), the power to inspect which users and groups can manipulate
resources in his namespace.

ResourceAccessReview is a runtime.Object with associated RESTStorage that only
accepts creates. The caller POSTs a ResourceAccessReview to this URL and he gets
a ResourceAccessReviewResponse back. Here is an example of a call and its
corresponding return:

```
// input
{
  "kind": "ResourceAccessReview",
  "apiVersion": "authorization.kubernetes.io/v1",
  "authorizationAttributes": {
    "verb": "list",
    "resource": "replicationcontrollers"
  }
}

// POSTed like this
curl -X POST /apis/authorization.kubernetes.io/{version}/resourceAccessReviews -d @resource-access-review.json
// or 
accessReviewResult, err := Client.ResourceAccessReviews().Create(resourceAccessReviewObject)

// output
{
  "kind": "ResourceAccessReviewResponse",
  "apiVersion": "authorization.kubernetes.io/v1",
  "namespace": "default"
  "users": ["Clark", "Hubert"],
  "groups": ["cluster-admins"]
}
```

The actual Go objects look like this:

```go
// ResourceAccessReview is a means to request a list of which users and groups
// are authorized to perform the action specified by spec
type ResourceAccessReview struct {
  kapi.TypeMeta `json:",inline"`

  // AuthorizationAttributes describes the action being tested.
  AuthorizationAttributes `json:"authorizationAttributes" description:"the action being tested"`
}

// ResourceAccessReviewResponse describes who can perform the action
type ResourceAccessReviewResponse struct {
  kapi.TypeMeta

  // Users is the list of users who can perform the action
  Users []string
  // Groups is the list of groups who can perform the action
  Groups []string
}

// LocalResourceAccessReview is a means to request a list of which users and
// groups are authorized to perform the action specified in a specific namespace
type LocalResourceAccessReview struct {
  kapi.TypeMeta `json:",inline"`

  // AuthorizationAttributes describes the action being tested.
  AuthorizationAttributes `json:"authorizationAttributes" description:"the action being tested"`
}

// LocalResourceAccessReviewResponse describes who can perform the action
type LocalResourceAccessReviewResponse struct {
  kapi.TypeMeta

  // Namespace is the namespace used for the access review
  Namespace string
  // Users is the list of users who can perform the action
  Users []string
  // Groups is the list of groups who can perform the action
  Groups []string
}
```




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/enhance-pluggable-policy.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
