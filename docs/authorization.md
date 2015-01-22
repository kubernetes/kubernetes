# Authorization Plugins


In Kubernetes, authorization happens as a separate step from authentication.
See the [authentication documentation](./authentication.md) for an 
overview of authentication.

Authorization applies to all HTTP accesses on the main apiserver port. (The
readonly port is not currently subject to authorization, but is planned to be
removed soon.)

The authorization check for any request compares attributes of the context of
the request, (such as user, resource kind, and namespace) with access
policies.  An API call must be allowed by some policy in order to proceed.

The following implementations are available, and are selected by flag:
  - `--authorization_mode=AlwaysDeny`
  - `--authorization_mode=AlwaysAllow`
  - `--authorization_mode=ABAC`

`AlwaysDeny` blocks all requests (used in tests).
`AlwaysAllow` allows all requests; use if you don't need authorization.
`ABAC` allows for user-configured authorization policy.  ABAC stands for Attribute-Based Access Control.

## ABAC Mode
### Request Attributes

A request has 4 attributes that can be considered for authorization:
  - user (the user-string which a user was authenticated as).
  - whether the request is readonly (GETs are readonly)
  - what kind of object is being accessed 
    - applies only to the API endpoints, such as 
        `/api/v1beta1/pods`.  For miscelaneous endpoints, like `/version`, the
        kind is the empty string.
  - the namespace of the object being access, or the empty string if the
        endpoint does not support namespaced objects.

We anticipate adding more attributes to allow finer grained access control and
to assist in policy management.

### Policy File Format

For mode `ABAC`, also specify `--authorization_policy_file=SOME_FILENAME`.

The file format is [one JSON object per line](http://jsonlines.org/).  There should be no enclosing list or map, just
one map per line.

Each line is a "policy object".  A policy object is a map with the following properties:
    - `user`, type string; the user-string from `--token_auth_file`
    - `readonly`, type boolean, when true, means that the policy only applies to GET
      operations.
    - `kind`, type string; a kind of object, from an URL, such as `pods`.
    - `namespace`, type string; a namespace string.

An unset property is the same as a property set to the zero value for its type (e.g. empty string, 0, false).
However, unset should be preferred for readability.

In the future, policies may be expressed in a JSON format, and managed via a REST
interface.

### Authorization Algorithm

A request has attributes which correspond to the properties of a policy object.

When a request is received, the attributes are determined.  Unknown attributes
are set to the zero value of its type (e.g. empty string, 0, false). 

An unset property will match any value of the corresponding
attribute.  An unset attribute will match any value of the corresponding property.

The tuple of attributes is checked for a match against every policy in the policy file.
If at least one line matches the request attributes, then the request is authorized (but may fail later validation).

To permit any user to do something, write a policy with the user property unset.
To permit an action Policy with an unset namespace applies regardless of namespace.

### Examples
 1. Alice can do anything: `{"user":"alice"}`
 2. Kubelet can read any pods: `{"user":"kubelet", "kind": "pods", "readonly": true}`
 3. Kubelet can read and write events: `{"user":"kubelet", "kind": "events"}`
 4. Bob can just read pods in namespace "projectCaribou": `{"user":"bob", "kind": "pods", "readonly": true, "ns": "projectCaribou"}`

[Complete file example](../pkg/auth/authorizer/abac/example_policy_file.jsonl)

## Plugin Developement

Other implementations can be developed fairly easily.
The APIserver calls the Authorizer interface:
```go
type Authorizer interface {
  Authorize(a Attributes) error
}
```
to determine whether or not to allow each API action.

An authorization plugin is a module that implements this interface.
Authorization plugin code goes in `pkg/auth/authorization/$MODULENAME`.

An authorization module can be completely implemented in go, or can call out
to a remote authorization service.  Authorization modules can implement
their own caching to reduce the cost of repeated authorization calls with the
same or similar arguments.  Developers should then consider the interaction between
caching and revocation of permissions.

## External APIs

### /api/{version}/ns/{namespace}/resourceAccessReview
This API answers the question: which users and groups can perform the specified verb on the specified resourceKind.

ResourceAccessReview is a runtime.Object with associated RESTStorage that only accepts creates.  The caller POSTs a ResourceAccessReview to this URL with the `spec` values filled in.  He gets a ResourceAccessReview back, with the `status` values completed.  Here is an example of a call and its corresponding return.
```
// input
{
  "kind": "ResourceAccessReview",
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "list-replicationController-check",
    "namespace": "default"
    },
  "spec": {
    "verb": "list",
    "resourceKind": "replicationControllers"
  }
}

// POSTed like this
curl -X POST /api/{version}/ns/{namespace}/resourceAccessReviews -d @resource-access-review.json
// or 
accessReviewResult, err := Client.ResourceAccessReviews(namespace).Create(resourceAccessReviewObject)

// output
{
  "kind": "ResourceAccessReview",
  "apiVersion": "v1beta3",
  "metadata": {
    "name": "list-replicationController-check",
    "namespace": "default"
    },
  "spec": {
    "verb": "list",
    "resourceKind": "replicationControllers"
  },
  "status": {
    "users": ["Clark", "Hubert"],
    "groups": ["cluster-admins"]
  }
}
```

The actual Go objects look like this:
```
type ResourceAccessReviewSpec struct{
  // Verb is one of: get, list, watch, create, update, delete
  Verb string

  // ResourceKind is one of the existing resource types
  ResourceKind string

  // Content is the actual content of the request for create and update
  Content runtime.EmbeddedObject

  // ResourceName is the name of the resource being requested for a "get" or deleted for a "delete"
  ResourceName string
}

type ResourceAccessReviewStatus struct{
  // Users is the list of users who can perform the action
  Users []string

  // Groups is the list of groups who can perform the action
  Groups []string

  // EvaluationError is optional.  It indicates why a ResourceAccessReview failed during evaluation
  EvaluationError string
}

type ResourceAccessReview struct {
  kapi.TypeMeta
  kapi.ObjectMeta

  Spec    ResourceAccessReviewSpec
  Status  ResourceAccessReviewStatus
}
```


