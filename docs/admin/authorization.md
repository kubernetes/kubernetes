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

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/admin/authorization.md).

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
  - what resource is being accessed
    - applies only to the API endpoints, such as
        `/api/v1/namespaces/default/pods`.  For miscellaneous endpoints, like `/version`, the
        resource is the empty string.
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
  - `resource`, type string; a resource from an URL, such as `pods`.
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
 2. Kubelet can read any pods: `{"user":"kubelet", "resource": "pods", "readonly": true}`
 3. Kubelet can read and write events: `{"user":"kubelet", "resource": "events"}`
 4. Bob can just read pods in namespace "projectCaribou": `{"user":"bob", "resource": "pods", "readonly": true, "ns": "projectCaribou"}`

[Complete file example](http://releases.k8s.io/HEAD/pkg/auth/authorizer/abac/example_policy_file.jsonl)

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
{"user":"system:serviceaccount:kube-system:default"}
```

The apiserver will need to be restarted to pickup the new policy lines.

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
Authorization plugin code goes in `pkg/auth/authorization/$MODULENAME`.

An authorization module can be completely implemented in go, or can call out
to a remote authorization service.  Authorization modules can implement
their own caching to reduce the cost of repeated authorization calls with the
same or similar arguments.  Developers should then consider the interaction between
caching and revocation of permissions.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/authorization.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
