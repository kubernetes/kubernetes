# Enhance Pluggable Policy
While trying to develop an authorization plugin for Kubernetes, we found a few places where API extensions would ease development and add power.  There are a few goals:
 1.  Provide an authorization plugin that can evaluate a .Authorize() call based on the full content of the request to RESTStorage.  This includes information like the full verb, the content of creates and updates, and the names of resources being acted upon.
 1.  Provide a mechanism for a user to determine whether or not he can take an action without attempting the action itself.
 1.  Provide a way for disconnected pieces of the system to ask whether a user is permitted to take an action without running in process with the API Authorizer.  For instance, a proxy for exec calls could ask whether a user can run the exec they are requesting.
 1.  Provide a mechanism to discover who can perform a given action on a given resource.  This is useful for answering questions like, "who can create replication controllers in my namespace".

This proposal adds to and extends existing API to so that authorizers may provide the functionality described above.  It does not attempt to describe how the policies themselves can be expressed, that is up the authorization plugins themselves.


## Enhancements to existing Authorization interfaces
The existing Authorization interfaces are described here: [authorization.md](../authorization.md).  A couple additions will allow the development of an Authorizer that matches based on different rules than the existing implementation.

### Request Attributes
The existing authorizer.Attributes only has 5 attributes (user, groups, isReadOnly, kind, and namespace).  If we add more detailed verbs, content, and resource names, then Authorizer plugins will have the same level of information available to RESTStorage components in order to express more detailed policy.  The replacement excerpt is below.

A request has 7 attributes that can be considered for authorization:
  - user - the user-string which a user was authenticated as.  This is included in the Context.
  - groups - the groups to which the user belongs.  This is included in the Context.
  - verb - string describing the requestion action. Today we have: get, list, watch, create, update, and delete
  - what kind of object is being accessed 
    - applies only to the API endpoints, such as 
        `/api/v1beta1/pods`.  For miscelaneous endpoints, like `/version`, the kind is the empty string.
  - the namespace of the object being access, or the empty string if the endpoint does not support namespaced objects.  This is included in the Context.
  - content - the runtime.Object being submitted. This only has content for creates and updates
  - resourceName - the name of the resource being gotten (get-ed) or deleted.

### Authorizer Interface
The existing Authorizer interface is very simple, but there isn't a way to provide details about allows, denies, or failures.  The extended detail is useful for UIs that want to describe why certain actions are disallowed.  Not all Authorizers will want to provide that information, but for those that do having that capability is useful.  In addition, adding a `GetAllowedSubjects` method that returns back the users and groups that can perform a particular action makes it possible to answer questions like, "who can see resources in my namespace" (see [ResourceAccessReview](#ResourceAccessReview) further down).

```
// OLD
type Authorizer interface {
  Authorize(a Attributes) error 
}
```

```
// NEW
type Authorizer interface {
  // Authorize takes a Context (for namespace, user, and traceability) and Attributes to make a policy determination.
  // reason is an optional return value that can describe why a policy decision was made.
  Authorize(ctx api.Context, a Attributes) (allowed bool, reason string, evaluationError error)

  // GetAllowedSubjects takes a Context (for namespace and traceability) and Attributes to determine which users and 
  // groups are allowed to perform the described action in the namespace.    This API enables the ResourceBasedReview requests below
  GetAllowedSubjects(ctx api.Context, a Attributes) (users util.StringSet, groups util.StringSet, evaluationError error)
}
```

### SubjectAccessReview
`/api/{version}/ns/{namespace}/subjectAccessReview` - This API answers the question: can a user or group (use authenticated user if none is specified) perform a given action.  Given the Authorizer interface (proposed or existing), this endpoint can be implemented generically against any Authorizer by creating the correct Attributes and making an .Authorize() call.

SubjectAccessReview is runtime.Object with associated RESTStorage that only accepts creates.  The caller POSTs a SubjectAccessReview to this URL and he gets a SubjectAccessReviewResponse back.  Here is an example of a call and its corresponding return.
```
// input
{
  "kind": "SubjectAccessReview",
  "apiVersion": "v1beta3",
  "verb": "create",
  "resource": "pods",
  "user": "Clark",
  "content": {
    "kind": "pods",
    "apiVersion": "v1beta3"
    // rest of pod content
  }
}

// POSTed like this
curl -X POST /api/{version}/ns/{namespace}/subjectAccessReviews -d @subject-access-review.json
// or 
accessReviewResult, err := Client.SubjectAccessReviews(namespace).Create(subjectAccessReviewObject)

// output
{
  "kind": "SubjectAccessReviewResponse",
  "apiVersion": "v1beta3",
  "namespace": "default",
  "allowed": true
}
```

The actual Go objects look like this:
```
// SubjectAccessReview is an object for requesting information about whether a user or group can perform an action
type SubjectAccessReview struct {
  kapi.TypeMeta

  // Verb is one of: get, list, watch, create, update, delete
  Verb string
  // Resource is one of the existing resource types
  Resource string
  // User is optional.  If both User and Groups are empty, the current authenticated user is used.
  User string
  // Groups is optional.  Groups is the list of groups to which the User belongs.
  Groups util.StringSet
  // Content is the actual content of the request for create and update
  Content runtime.EmbeddedObject
  // ResourceName is the name of the resource being requested for a "get" or deleted for a "delete"
  ResourceName string
}

// SubjectAccessReviewResponse describes whether or not a user or group can perform an action
type SubjectAccessReviewResponse struct {
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
`/api/{version}/ns/{namespace}/resourceAccessReview` - This API answers the question: which users and groups can perform the specified verb on the specified resourceKind.  Given the Authorizer interface described above, this endpoint can be implemented generically against any Authorizer by calling the .GetAllowedSubjects() function.

ResourceAccessReview is a runtime.Object with associated RESTStorage that only accepts creates.  The caller POSTs a ResourceAccessReview to this URL and he gets a ResourceAccessReviewResponse back.  Here is an example of a call and its corresponding return.
```
// input
{
  "kind": "ResourceAccessReview",
  "apiVersion": "v1beta3",
  "verb": "list",
  "resource": "replicationcontrollers"
}

// POSTed like this
curl -X POST /api/{version}/ns/{namespace}/resourceAccessReviews -d @resource-access-review.json
// or 
accessReviewResult, err := Client.ResourceAccessReviews(namespace).Create(resourceAccessReviewObject)

// output
{
  "kind": "ResourceAccessReviewResponse",
  "apiVersion": "v1beta3",
  "namespace": "default"
  "users": ["Clark", "Hubert"],
  "groups": ["cluster-admins"]
}
```

The actual Go objects look like this:
```
// ResourceAccessReview is a means to request a list of which users and groups are authorized to perform the
// action specified by spec
type ResourceAccessReview struct {
  kapi.TypeMeta

  // Verb is one of: get, list, watch, create, update, delete
  Verb string
  // Resource is one of the existing resource types
  Resource string
  // Content is the actual content of the request for create and update
  Content runtime.EmbeddedObject
  // ResourceName is the name of the resource being requested for a "get" or deleted for a "delete"
  ResourceName string
}

// ResourceAccessReviewResponse describes who can perform the action
type ResourceAccessReviewResponse struct {
  kapi.TypeMeta

  // Namespace is the namespace used for the access review
  Namespace string
  // Users is the list of users who can perform the action
  Users util.StringSet
  // Groups is the list of groups who can perform the action
  Groups util.StringSet
}
```
