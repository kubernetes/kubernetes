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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/admin/authorization.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Authorization Plugins


In Kubernetes, authorization happens as a separate step from authentication.
See the [authentication documentation](authentication.md) for an
overview of authentication.

Authorization applies to all HTTP accesses on the main (secure) apiserver port.

The authorization check for any request compares attributes of the context of
the request, (such as user, resource, and namespace) with access
policies.  An API call must be allowed by some policy in order to proceed.

The following implementations are available, and are selected by flag:
  - `--authorization-mode=AlwaysDeny`
  - `--authorization-mode=AlwaysAllow`
  - `--authorization-mode=ABAC`
  - `--authorization-mode=Webhook`

`AlwaysDeny` blocks all requests (used in tests).
`AlwaysAllow` allows all requests; use if you don't need authorization.
`ABAC` allows for user-configured authorization policy.  ABAC stands for Attribute-Based Access Control.
`Webhook` allows for authorization to be driven by a remote service using REST.

## ABAC Mode

### Request Attributes

A request has the following attributes that can be considered for authorization:
  - user (the user-string which a user was authenticated as).
  - group (the list of group names the authenticated user is a member of).
  - whether the request is for an API resource.
  - the request path.
    - allows authorizing access to miscellaneous endpoints like `/api` or `/healthz` (see [kubectl](#kubectl)).
  - the request verb.
    - API verbs like `get`, `list`, `create`, `update`, `watch`, `delete`, and `deletecollection` are used for API requests
    - HTTP verbs like `get`, `post`, `put`, and `delete` are used for non-API requests
  - what resource is being accessed (for API requests only)
  - the namespace of the object being accessed (for namespaced API requests only)
  - the API group being accessed (for API requests only)

We anticipate adding more attributes to allow finer grained access control and
to assist in policy management.

### Policy File Format

For mode `ABAC`, also specify `--authorization-policy-file=SOME_FILENAME`.

The file format is [one JSON object per line](http://jsonlines.org/).  There should be no enclosing list or map, just
one map per line.

Each line is a "policy object".  A policy object is a map with the following properties:
  - Versioning properties:
    - `apiVersion`, type string; valid values are "abac.authorization.kubernetes.io/v1beta1". Allows versioning and conversion of the policy format.
    - `kind`, type string: valid values are "Policy". Allows versioning and conversion of the policy format.

  - `spec` property set to a map with the following properties:
    - Subject-matching properties:
      - `user`, type string; the user-string from `--token-auth-file`. If you specify `user`, it must match the username of the authenticated user. `*` matches all requests.
      - `group`, type string; if you specify `group`, it must match one of the groups of the authenticated user. `*` matches all requests.

    - `readonly`, type boolean, when true, means that the policy only applies to get, list, and watch operations.

    - Resource-matching properties:
      - `apiGroup`, type string; an API group, such as `extensions`. `*` matches all API groups.
      - `namespace`, type string; a namespace string. `*` matches all resource requests.
      - `resource`, type string; a resource, such as `pods`. `*` matches all resource requests.

    - Non-resource-matching properties:
    - `nonResourcePath`, type string; matches the non-resource request paths (like `/version` and `/apis`). `*` matches all non-resource requests. `/foo/*` matches `/foo/` and all of its subpaths.

An unset property is the same as a property set to the zero value for its type (e.g. empty string, 0, false).
However, unset should be preferred for readability.

In the future, policies may be expressed in a JSON format, and managed via a REST interface.

### Authorization Algorithm

A request has attributes which correspond to the properties of a policy object.

When a request is received, the attributes are determined.  Unknown attributes
are set to the zero value of its type (e.g. empty string, 0, false).

A property set to "*" will match any value of the corresponding attribute.

The tuple of attributes is checked for a match against every policy in the policy file.
If at least one line matches the request attributes, then the request is authorized (but may fail later validation).

To permit any user to do something, write a policy with the user property set to "*".
To permit a user to do anything, write a policy with the apiGroup, namespace, resource, and nonResourcePath properties set to "*".

### Kubectl

Kubectl uses the `/api` and `/apis` endpoints of api-server to negotiate client/server versions. To validate objects sent to the API by create/update operations, kubectl queries certain swagger resources. For API version `v1` those would be `/swaggerapi/api/v1` & `/swaggerapi/experimental/v1`.

When using ABAC authorization, those special resources have to be explicitly exposed via the `nonResourcePath` property in a policy (see [examples](#examples) below):

* `/api`, `/api/*`, `/apis`, and `/apis/*` for API version negotiation.
* `/version` for retrieving the server version via `kubectl version`.
* `/swaggerapi/*` for create/update operations.

To inspect the HTTP calls involved in a specific kubectl operation you can turn up the verbosity:

    kubectl --v=8 version

### Examples

 1. Alice can do anything to all resources:                  `{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user": "alice", "namespace": "*", "resource": "*", "apiGroup": "*"}}`
 2. Kim can read any pods:                               `{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user": "kim", "namespace": "*", "resource": "pods", "readonly": true}}`
 3. Kim can read and write events:                       `{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user": "kim", "namespace": "*", "resource": "events"}}`
 4. Bob can just read pods in namespace "projectCaribou":    `{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user": "bob", "namespace": "projectCaribou", "resource": "pods", "readonly": true}}`
 5. Anyone can make read-only requests to all non-API paths: `{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user": "*", "readonly": true, "nonResourcePath": "*"}}`

[Complete file example](http://releases.k8s.io/HEAD/pkg/auth/authorizer/abac/example_policy_file.jsonl)

#### In Practice

If we assume a very simple setup where we have 3 users:
 - `admin` which are external users of API (`kubectl`, third party applications, etc).
 - `kubelet` which is the user for all `kubelet`s in the cluster.
 - `scheduler` which is the user for the `kube-scheduler` within the cluster.

And we want to allow all other users to have read only access across all of the resources.

We would want a policy file like:
```
{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user":"*", "apiGroup": "*", "nonResourcePath": "*", "resource":"*", "readonly": true }}
{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user":"admin", "apiGroup": "*", "namespace": "*", "resource": "*", "readonly": false }}
{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user":"scheduler", "apiGroup": "*", "namespace": "*", "resource": "*", "readonly": false }}
{"apiVersion": "abac.authorization.kubernetes.io/v1beta1", "kind": "Policy", "spec": {"user":"kubelet", "apiGroup": "*", "namespace": "*", "resource": "*", "readonly": false }}
```


### A quick note on service accounts

A service account automatically generates a user. The user's name is generated according to the naming convention:

```
system:serviceaccount:<namespace>:<serviceaccountname>
```

Creating a new namespace also causes a new service account to be created, of this form:*

```
system:serviceaccount:<namespace>:default
```

For example, if you wanted to grant the default service account in the kube-system full privilege to the API, you would add this line to your policy file:

```json
{"apiVersion":"abac.authorization.kubernetes.io/v1beta1","kind":"Policy","user":"system:serviceaccount:kube-system:default","namespace":"*","resource":"*","apiGroup":"*"}
```

The apiserver will need to be restarted to pickup the new policy lines.

## Webhook Mode

When specified, mode `Webhook` causes Kubernetes to query an outside REST service when determining user privileges.

### Configuration File Format

Mode `Webhook` requires a file for HTTP configuration, specify by the `--authorization-webhook-config-file=SOME_FILENAME` flag.

The configuration file uses the [kubeconfig](../user-guide/kubeconfig-file.md) file format. Within the file "users" refers to the API Server webhook and "clusters" refers to the remote service.

A configuration example which uses HTTPS client auth:

```yaml
# clusters refers to the remote service.
clusters:
  - name: name-of-remote-authz-service
    cluster:
      certificate-authority: /path/to/ca.pem      # CA for verifying the remote service.
      server: https://authz.example.com/authorize # URL of remote service to query. Must use 'https'.

# users refers to the API Server's webhook configuration.
users:
  - name: name-of-api-server
    user:
      client-certificate: /path/to/cert.pem # cert for the webhook plugin to use
      client-key: /path/to/key.pem          # key matching the cert
```

### Request Payloads

When faced with an authorization decision, the API Server POSTs a JSON serialized api.authorization.v1beta1.SubjectAccessReview object describing the action. This object contains fields describing the user attempting to make the request, and either details about the resource being accessed or requests attributes.

Note that webhook API objects are subject to the same [versioning compatibility rules](../api.md) as other Kubernetes API objects. Implementers should be aware of loser compatibility promises for beta objects and check the "apiVersion" field of the request to ensure correct deserialization. Additionally, the API Server must enable the `authorization.k8s.io/v1beta1` API extensions group (`--runtime-config=authorization.k8s.io/v1beta1=true`).

An example request body:

```json
{
  "apiVersion": "authorization.k8s.io/v1beta1",
  "kind": "SubjectAccessReview",
  "spec": {
    "resourceAttributes": {
      "namespace": "kittensandponies",
      "verb": "GET",
      "group": "*",
      "resource": "pods"
    },
    "user": "jane",
    "group": [
      "group1",
      "group2"
    ]
  }
}
```

The remote service is expected to fill the SubjectAccessReviewStatus field of the request and respond to either allow or disallow access. The response body's "spec" field is ignored and may be omitted. A permissive response would return:

```json
{
  "apiVersion": "authorization.k8s.io/v1beta1",
  "kind": "SubjectAccessReview",
  "status": {
    "allowed": true
  }
}
```

To disallow access, the remote service would return:

```json
{
  "apiVersion": "authorization.k8s.io/v1beta1",
  "kind": "SubjectAccessReview",
  "status": {
    "allowed": false,
    "reason": "user does not have read access to the namespace"
  }
}
```

Access to non-resource paths are sent as:

```json
{
  "apiVersion": "authorization.k8s.io/v1beta1",
  "kind": "SubjectAccessReview",
  "spec": {
    "nonResourceAttributes": {
      "path": "/debug",
      "verb": "GET"
    },
    "user": "jane",
    "group": [
      "group1",
      "group2"
    ]
  }
}
```

Non-resource paths include: `/api`, `/apis`, `/metrics`, `/resetMetrics`, `/logs`, `/debug`, `/healthz`, `/swagger-ui/`, `/swaggerapi/`, `/ui`, and `/version.` Clients require access to `/api`, `/api/*/`, `/apis/`, `/apis/*`, `/apis/*/*`, and `/version` to discover what resources and versions are present on the server. Access to other non-resource paths can be disallowed without restricting access to the REST api.

For further documentation refer to the authorization.v1beta1 API objects and plugin/pkg/auth/authorizer/webhook/webhook.go.

## Plugin Development

Other implementations can be developed fairly easily.
The APIserver calls the Authorizer interface:

```go
type Authorizer interface {
  Authorize(a Attributes) error
}
```

to determine whether or not to allow each API action.

An authorization plugin is a module that implements this interface.
Authorization plugin code goes in `pkg/auth/authorizer/$MODULENAME`.

An authorization module can be completely implemented in go, or can call out
to a remote authorization service.  Authorization modules can implement
their own caching to reduce the cost of repeated authorization calls with the
same or similar arguments.  Developers should then consider the interaction between
caching and revocation of permissions.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/authorization.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
