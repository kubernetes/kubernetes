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
[here](http://releases.k8s.io/release-1.0/docs/devel/api-conventions.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
API Conventions
===============

Updated: 4/16/2015

*This document is oriented at users who want a deeper understanding of the Kubernetes
API structure, and developers wanting to extend the Kubernetes API.  An introduction to
using resources with kubectl can be found in (working_with_resources.md).*

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

  - [Types (Kinds)](#types-kinds)
    - [Resources](#resources)
    - [Objects](#objects)
      - [Metadata](#metadata)
      - [Spec and Status](#spec-and-status)
        - [Typical status properties](#typical-status-properties)
      - [References to related objects](#references-to-related-objects)
      - [Lists of named subobjects preferred over maps](#lists-of-named-subobjects-preferred-over-maps)
      - [Constants](#constants)
    - [Lists and Simple kinds](#lists-and-simple-kinds)
  - [Differing Representations](#differing-representations)
  - [Verbs on Resources](#verbs-on-resources)
    - [PATCH operations](#patch-operations)
      - [Strategic Merge Patch](#strategic-merge-patch)
    - [List Operations](#list-operations)
    - [Map Operations](#map-operations)
  - [Idempotency](#idempotency)
  - [Defaulting](#defaulting)
  - [Late Initialization](#late-initialization)
  - [Concurrency Control and Consistency](#concurrency-control-and-consistency)
  - [Serialization Format](#serialization-format)
  - [Units](#units)
  - [Selecting Fields](#selecting-fields)
  - [HTTP Status codes](#http-status-codes)
      - [Success codes](#success-codes)
      - [Error codes](#error-codes)
  - [Response Status Kind](#response-status-kind)
  - [Events](#events)

<!-- END MUNGE: GENERATED_TOC -->

The conventions of the [Kubernetes API](../api.md) (and related APIs in the ecosystem) are intended to ease client development and ensure that configuration mechanisms can be implemented that work across a diverse set of use cases consistently.

The general style of the Kubernetes API is RESTful - clients create, update, delete, or retrieve a description of an object via the standard HTTP verbs (POST, PUT, DELETE, and GET) - and those APIs preferentially accept and return JSON. Kubernetes also exposes additional endpoints for non-standard verbs and allows alternative content types. All of the JSON accepted and returned by the server has a schema, identified by the "kind" and "apiVersion" fields. Where relevant HTTP header fields exist, they should mirror the content of JSON fields, but the information should not be represented only in the HTTP header.

The following terms are defined:

* **Kind** the name of a particular object schema (e.g. the "Cat" and "Dog" kinds would have different attributes and properties)
* **Resource** a representation of a system entity, sent or retrieved as JSON via HTTP to the server. Resources are exposed via:
  * Collections - a list of resources of the same type, which may be queryable
  * Elements - an individual resource, addressable via a URL

Each resource typically accepts and returns data of a single kind.  A kind may be accepted or returned by multiple resources that reflect specific use cases. For instance, the kind "pod" is exposed as a "pods" resource that allows end users to create, update, and delete pods, while a separate "pod status" resource (that acts on "pod" kind) allows automated processes to update a subset of the fields in that resource. A "restart" resource might be exposed for a number of different resources to allow the same action to have different results for each object.

Resource collections should be all lowercase and plural, whereas kinds are CamelCase and singular.


## Types (Kinds)

Kinds are grouped into three categories:

1. **Objects** represent a persistent entity in the system.

   Creating an API object is a record of intent - once created, the system will work to ensure that resource exists. All API objects have common metadata.

   An object may have multiple resources that clients can use to perform specific actions that create, update, delete, or get.

   Examples: `Pods`, `ReplicationControllers`, `Services`, `Namespaces`, `Nodes`

2. **Lists** are collections of **resources** of one (usually) or more (occasionally) kinds.

   Lists have a limited set of common metadata. All lists use the "items" field to contain the array of objects they return.

   Most objects defined in the system should have an endpoint that returns the full set of resources, as well as zero or more endpoints that return subsets of the full list. Some objects may be singletons (the current user, the system defaults) and may not have lists.

   In addition, all lists that return objects with labels should support label filtering (see [docs/user-guide/labels.md](../user-guide/labels.md), and most lists should support filtering by fields.

   Examples: PodLists, ServiceLists, NodeLists

   TODO: Describe field filtering below or in a separate doc.

3. **Simple** kinds are used for specific actions on objects and for non-persistent entities.

   Given their limited scope, they have the same set of limited common metadata as lists.

   The "size" action may accept a simple resource that has only a single field as input (the number of things). The "status" kind is returned when errors occur and is not persisted in the system.

   Examples: Binding, Status

The standard REST verbs (defined below) MUST return singular JSON objects. Some API endpoints may deviate from the strict REST pattern and return resources that are not singular JSON objects, such as streams of JSON objects or unstructured text log data.

The term "kind" is reserved for these "top-level" API types. The term "type" should be used for distinguishing sub-categories within objects or subobjects.

### Resources

All JSON objects returned by an API MUST have the following fields:

* kind: a string that identifies the schema this object should have
* apiVersion: a string that identifies the version of the schema the object should have

These fields are required for proper decoding of the object. They may be populated by the server by default from the specified URL path, but the client likely needs to know the values in order to construct the URL path.

### Objects

#### Metadata

Every object kind MUST have the following metadata in a nested object field called "metadata":

* namespace: a namespace is a DNS compatible subdomain that objects are subdivided into. The default namespace is 'default'.  See [docs/user-guide/namespaces.md](../user-guide/namespaces.md) for more.
* name: a string that uniquely identifies this object within the current namespace (see [docs/user-guide/identifiers.md](../user-guide/identifiers.md)). This value is used in the path when retrieving an individual object.
* uid: a unique in time and space value (typically an RFC 4122 generated identifier, see [docs/user-guide/identifiers.md](../user-guide/identifiers.md)) used to distinguish between objects with the same name that have been deleted and recreated

Every object SHOULD have the following metadata in a nested object field called "metadata":

* resourceVersion: a string that identifies the internal version of this object that can be used by clients to determine when objects have changed. This value MUST be treated as opaque by clients and passed unmodified back to the server. Clients should not assume that the resource version has meaning across namespaces, different kinds of resources, or different servers. (see [concurrency control](#concurrency-control-and-consistency), below, for more details)
* creationTimestamp: a string representing an RFC 3339 date of the date and time an object was created
* deletionTimestamp: a string representing an RFC 3339 date of the date and time after which this resource will be deleted. This field is set by the server when a graceful deletion is requested by the user, and is not directly settable by a client. The resource will be deleted (no longer visible from resource lists, and not reachable by name) after the time in this field. Once set, this value may not be unset or be set further into the future, although it may be shortened or the resource may be deleted prior to this time.
* labels: a map of string keys and values that can be used to organize and categorize objects (see [docs/user-guide/labels.md](../user-guide/labels.md))
* annotations: a map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about this object (see [docs/user-guide/annotations.md](../user-guide/annotations.md))

Labels are intended for organizational purposes by end users (select the pods that match this label query). Annotations enable third-party automation and tooling to decorate objects with additional metadata for their own use.

#### Spec and Status

By convention, the Kubernetes API makes a distinction between the specification of the desired state of an object (a nested object field called "spec") and the status of the object at the current time (a nested object field called "status"). The specification is a complete description of the desired state, including configuration settings provided by the user, [default values](#defaulting) expanded by the system, and properties initialized or otherwise changed after creation by other ecosystem components (e.g., schedulers, auto-scalers), and is persisted in stable storage with the API object. If the specification is deleted, the object will be purged from the system. The status summarizes the current state of the object in the system, and is usually persisted with the object by an automated processes but may be generated on the fly. At some cost and perhaps some temporary degradation in behavior, the status could be reconstructed by observation if it were lost.

When a new version of an object is POSTed or PUT, the "spec" is updated and available immediately. Over time the system will work to bring the "status" into line with the "spec". The system will drive toward the most recent "spec" regardless of previous versions of that stanza. In other words, if a value is changed from 2 to 5 in one PUT and then back down to 3 in another PUT the system is not required to 'touch base' at 5 before changing the "status" to 3. In other words, the system's behavior is *level-based* rather than *edge-based*. This enables robust behavior in the presence of missed intermediate state changes.

The Kubernetes API also serves as the foundation for the declarative configuration schema for the system. In order to facilitate level-based operation and expression of declarative configuration, fields in the specification should have declarative rather than imperative names and semantics -- they represent the desired state, not actions intended to yield the desired state.

The PUT and POST verbs on objects will ignore the "status" values. A `/status` subresource is provided to enable system components to update statuses of resources they manage.

Otherwise, PUT expects the whole object to be specified. Therefore, if a field is omitted it is assumed that the client wants to clear that field's value. The PUT verb does not accept partial updates. Modification of just part of an object may be achieved by GETting the resource, modifying part of the spec, labels, or annotations, and then PUTting it back. See [concurrency control](#concurrency-control-and-consistency), below, regarding read-modify-write consistency when using this pattern. Some objects may expose alternative resource representations that allow mutation of the status, or performing custom actions on the object.

All objects that represent a physical resource whose state may vary from the user's desired intent SHOULD have a "spec" and a "status".  Objects whose state cannot vary from the user's desired intent MAY have only "spec", and MAY rename "spec" to a more appropriate name.

Objects that contain both spec and status should not contain additional top-level fields other than the standard metadata fields.

##### Typical status properties

* **phase**: The phase is a simple, high-level summary of the phase of the lifecycle of an object. The phase should progress monotonically. Typical phase values are `Pending` (not yet fully physically realized), `Running` or `Active` (fully realized and active, but not necessarily operating correctly), and `Terminated` (no longer active), but may vary slightly for different types of objects. New phase values should not be added to existing objects in the future. Like other status fields, it must be possible to ascertain the lifecycle phase by observation. Additional details regarding the current phase may be contained in other fields.
* **conditions**: Conditions represent orthogonal observations of an object's current state. Objects may report multiple conditions, and new types of conditions may be added in the future. Condition status values may be `True`, `False`, or `Unknown`. Unlike the phase, conditions are not expected to be monotonic -- their values may change back and forth. A typical condition type is `Ready`, which indicates the object was believed to be fully operational at the time it was last probed. Conditions may carry additional information, such as the last probe time or last transition time.

TODO(@vishh): Reason and Message.

Phases and conditions are observations and not, themselves, state machines, nor do we define comprehensive state machines for objects with behaviors associated with state transitions. The system is level-based and should assume an Open World. Additionally, new observations and details about these observations may be added over time.

In order to preserve extensibility, in the future, we intend to explicitly convey properties that users and components care about rather than requiring those properties to be inferred from observations.

Note that historical information status (e.g., last transition time, failure counts) is only provided at best effort, and is not guaranteed to not be lost.

Status information that may be large (especially unbounded in size, such as lists of references to other objects -- see below) and/or rapidly changing, such as [resource usage](../design/resources.md#usage-data), should be put into separate objects, with possibly a reference from the original object. This helps to ensure that GETs and watch remain reasonably efficient for the majority of clients, which may not need that data.

#### References to related objects

References to loosely coupled sets of objects, such as [pods](../user-guide/pods.md) overseen by a [replication controller](../user-guide/replication-controller.md), are usually best referred to using a [label selector](../user-guide/labels.md). In order to ensure that GETs of individual objects remain bounded in time and space, these sets may be queried via separate API queries, but will not be expanded in the referring object's status.

References to specific objects, especially specific resource versions and/or specific fields of those objects, are specified using the `ObjectReference` type. Unlike partial URLs, the ObjectReference type facilitates flexible defaulting of fields from the referring object or other contextual information.

References in the status of the referee to the referrer may be permitted, when the references are one-to-one and do not need to be frequently updated, particularly in an edge-based manner.

#### Lists of named subobjects preferred over maps

Discussed in [#2004](https://github.com/GoogleCloudPlatform/kubernetes/issues/2004) and elsewhere. There are no maps of subobjects in any API objects. Instead, the convention is to use a list of subobjects containing name fields.

For example:

```yaml
ports:
  - name: www
    containerPort: 80
```

vs.

```yaml
ports:
  www:
    containerPort: 80
```

This rule maintains the invariant that all JSON/YAML keys are fields in API objects. The only exceptions are pure maps in the API (currently, labels, selectors, and annotations), as opposed to sets of subobjects.

#### Constants

Some fields will have a list of allowed values (enumerations). These values will be strings, and they will be in CamelCase, with an initial uppercase letter. Examples: "ClusterFirst", "Pending", "ClientIP".

### Lists and Simple kinds

Every list or simple kind SHOULD have the following metadata in a nested object field called "metadata":

* resourceVersion: a string that identifies the common version of the objects returned by in a list. This value MUST be treated as opaque by clients and passed unmodified back to the server. A resource version is only valid within a single namespace on a single kind of resource.

Every simple kind returned by the server, and any simple kind sent to the server that must support idempotency or optimistic concurrency should return this value.Since simple resources are often used as input alternate actions that modify objects, the resource version of the simple resource should correspond to the resource version of the object.


## Differing Representations

An API may represent a single entity in different ways for different clients, or transform an object after certain transitions in the system occur. In these cases, one request object may have two representations available as different resources, or different kinds.

An example is a Service, which represents the intent of the user to group a set of pods with common behavior on common ports. When Kubernetes detects a pod matches the service selector, the IP address and port of the pod are added to an Endpoints resource for that Service. The Endpoints resource exists only if the Service exists, but exposes only the IPs and ports of the selected pods.  The full service is represented by two distinct resources - under the original Service resource the user created, as well as in the Endpoints resource.

As another example, a "pod status" resource may accept a PUT with the "pod" kind, with different rules about what fields may be changed.

Future versions of Kubernetes may allow alternative encodings of objects beyond JSON.


## Verbs on Resources

API resources should use the traditional REST pattern:

* GET /&lt;resourceNamePlural&gt; - Retrieve a list of type &lt;resourceName&gt;, e.g. GET /pods returns a list of Pods.
* POST /&lt;resourceNamePlural&gt; - Create a new resource from the JSON object provided by the client.
* GET /&lt;resourceNamePlural&gt;/&lt;name&gt; - Retrieves a single resource with the given name, e.g. GET /pods/first returns a Pod named 'first'. Should be constant time, and the resource should be bounded in size.
* DELETE /&lt;resourceNamePlural&gt;/&lt;name&gt;  - Delete the single resource with the given name. DeleteOptions may specify gracePeriodSeconds, the optional duration in seconds before the object should be deleted. Individual kinds may declare fields which provide a default grace period, and different kinds may have differing kind-wide default grace periods. A user provided grace period overrides a default grace period, including the zero grace period ("now").
* PUT /&lt;resourceNamePlural&gt;/&lt;name&gt; - Update or create the resource with the given name with the JSON object provided by the client.
* PATCH /&lt;resourceNamePlural&gt;/&lt;name&gt; - Selectively modify the specified fields of the resource. See more information [below](#patch).

Kubernetes by convention exposes additional verbs as new root endpoints with singular names. Examples:

* GET /watch/&lt;resourceNamePlural&gt; - Receive a stream of JSON objects corresponding to changes made to any resource of the given kind over time.
* GET /watch/&lt;resourceNamePlural&gt;/&lt;name&gt; - Receive a stream of JSON objects corresponding to changes made to the named resource of the given kind over time.

These are verbs which change the fundamental type of data returned (watch returns a stream of JSON instead of a single JSON object). Support of additional verbs is not required for all object types.

Two additional verbs `redirect` and `proxy` provide access to cluster resources as described in [docs/user-guide/accessing-the-cluster.md](../user-guide/accessing-the-cluster.md).

When resources wish to expose alternative actions that are closely coupled to a single resource, they should do so using new sub-resources. An example is allowing automated processes to update the "status" field of a Pod. The `/pods` endpoint only allows updates to "metadata" and "spec", since those reflect end-user intent. An automated process should be able to modify status for users to see by sending an updated Pod kind to the server to the "/pods/&lt;name&gt;/status" endpoint - the alternate endpoint allows different rules to be applied to the update, and access to be appropriately restricted. Likewise, some actions like "stop" or "scale" are best represented as REST sub-resources that are POSTed to.  The POST action may require a simple kind to be provided if the action requires parameters, or function without a request body.

TODO: more documentation of Watch

### PATCH operations

The API supports three different PATCH operations, determined by their corresponding Content-Type header:

* JSON Patch, `Content-Type: application/json-patch+json`
 * As defined in [RFC6902](https://tools.ietf.org/html/rfc6902), a JSON Patch is a sequence of operations that are executed on the resource, e.g. `{"op": "add", "path": "/a/b/c", "value": [ "foo", "bar" ]}`. For more details on how to use JSON Patch, see the RFC.
* Merge Patch, `Content-Type: application/merge-json-patch+json`
 * As defined in [RFC7386](https://tools.ietf.org/html/rfc7386), a Merge Patch is essentially a partial representation of the resource. The submitted JSON is "merged" with the current resource to create a new one, then the new one is saved. For more details on how to use Merge Patch, see the RFC.
* Strategic Merge Patch, `Content-Type: application/strategic-merge-patch+json`
 * Strategic Merge Patch is a custom implementation of Merge Patch. For a detailed explanation of how it works and why it needed to be introduced, see below.

#### Strategic Merge Patch

In the standard JSON merge patch, JSON objects are always merged but lists are always replaced. Often that isn't what we want. Let's say we start with the following Pod:

```yaml
spec:
  containers:
    - name: nginx
      image: nginx-1.0
```

...and we POST that to the server (as JSON). Then let's say we want to *add* a container to this Pod.

```yaml
PATCH /api/v1/namespaces/default/pods/pod-name
spec:
  containers:
    - name: log-tailer
      image: log-tailer-1.0
```

If we were to use standard Merge Patch, the entire container list would be replaced with the single log-tailer container. However, our intent is for the container lists to merge together based on the `name` field.

To solve this problem, Strategic Merge Patch uses metadata attached to the API objects to determine what lists should be merged and which ones should not. Currently the metadata is available as struct tags on the API objects themselves, but will become available to clients as Swagger annotations in the future. In the above example, the `patchStrategy` metadata for the `containers` field would be `merge` and the `patchMergeKey` would be `name`.

Note: If the patch results in merging two lists of scalars, the scalars are first deduplicated and then merged.

Strategic Merge Patch also supports special operations as listed below.

### List Operations

To override the container list to be strictly replaced, regardless of the default:

```yaml
containers:
  - name: nginx
    image: nginx-1.0
  - $patch: replace   # any further $patch operations nested in this list will be ignored
```

To delete an element of a list that should be merged:

```yaml
containers:
  - name: nginx
    image: nginx-1.0
  - $patch: delete
    name: log-tailer  # merge key and value goes here
```

### Map Operations

To indicate that a map should not be merged and instead should be taken literally:

```yaml
$patch: replace  # recursive and applies to all fields of the map it's in
containers:
- name: nginx
  image: nginx-1.0
```

To delete a field of a map:

```yaml
name: nginx
image: nginx-1.0
labels:
  live: null  # set the value of the map key to null
```


## Idempotency

All compatible Kubernetes APIs MUST support "name idempotency" and respond with an HTTP status code 409 when a request is made to POST an object that has the same name as an existing object in the system. See [docs/user-guide/identifiers.md](../user-guide/identifiers.md) for details.

Names generated by the system may be requested using `metadata.generateName`. GenerateName indicates that the name should be made unique by the server prior to persisting it. A non-empty value for the field indicates the name will be made unique (and the name returned to the client will be different than the name passed). The value of this field will be combined with a unique suffix on the server if the Name field has not been provided. The provided value must be valid within the rules for Name, and may be truncated by the length of the suffix required to make the value unique on the server. If this field is specified, and Name is not present, the server will NOT return a 409 if the generated name exists - instead, it will either return 201 Created or 504 with Reason `ServerTimeout` indicating a unique name could not be found in the time allotted, and the client should retry (optionally after the time indicated in the Retry-After header).

## Defaulting

Default resource values are API version-specific, and they are applied during
the conversion from API-versioned declarative configuration to internal objects
representing the desired state (`Spec`) of the resource. Subsequent GETs of the
resource will include the default values explicitly.

Incorporating the default values into the `Spec` ensures that `Spec` depicts the
full desired state so that it is easier for the system to determine how to
achieve the state, and for the user to know what to anticipate.

API version-specific default values are set by the API server.

## Late Initialization

Late initialization is when resource fields are set by a system controller
after an object is created/updated.

For example, the scheduler sets the `pod.spec.nodeName` field after the pod is created.

Late-initializers should only make the following types of modifications:
 - Setting previously unset fields
 - Adding keys to maps
 - Adding values to arrays which have mergeable semantics (`patchStrategy:"merge"` attribute in
  the type definition).

These conventions:
 1. allow a user (with sufficient privilege) to override any system-default behaviors by setting
    the fields that would otherwise have been defaulted.
 1. enables updates from users to be merged with changes made during late initialization, using
    strategic merge patch, as opposed to clobbering the change.
 1. allow the component which does the late-initialization to use strategic merge patch, which
    facilitates composition and concurrency of such components.

Although the apiserver Admission Control stage acts prior to object creation,
Admission Control plugins should follow the Late Initialization conventions
too, to allow their implementation to be later moved to a 'controller', or to client libraries.

## Concurrency Control and Consistency

Kubernetes leverages the concept of *resource versions* to achieve optimistic concurrency. All Kubernetes resources have a "resourceVersion" field as part of their metadata. This resourceVersion is a string that identifies the internal version of an object that can be used by clients to determine when objects have changed. When a record is about to be updated, it's version is checked against a pre-saved value, and if it doesn't match, the update fails with a StatusConflict (HTTP status code 409).

The resourceVersion is changed by the server every time an object is modified. If resourceVersion is included with the PUT operation the system will verify that there have not been other successful mutations to the resource during a read/modify/write cycle, by verifying that the current value of resourceVersion matches the specified value.

The resourceVersion is currently backed by [etcd's modifiedIndex](https://coreos.com/docs/distributed-configuration/etcd-api/). However, it's important to note that the application should *not* rely on the implementation details of the versioning system maintained by Kubernetes. We may change the implementation of resourceVersion in the future, such as to change it to a timestamp or per-object counter.

The only way for a client to know the expected value of resourceVersion is to have received it from the server in response to a prior operation, typically a GET. This value MUST be treated as opaque by clients and passed unmodified back to the server. Clients should not assume that the resource version has meaning across namespaces, different kinds of resources, or different servers. Currently, the value of resourceVersion is set to match etcd's sequencer. You could think of it as a logical clock the API server can use to order requests. However, we expect the implementation of resourceVersion to change in the future, such as in the case we shard the state by kind and/or namespace, or port to another storage system.

In the case of a conflict, the correct client action at this point is to GET the resource again, apply the changes afresh, and try submitting again. This mechanism can be used to prevent races like the following:

```
Client #1                                  Client #2
GET Foo                                    GET Foo
Set Foo.Bar = "one"                        Set Foo.Baz = "two"
PUT Foo                                    PUT Foo
```

When these sequences occur in parallel, either the change to Foo.Bar or the change to Foo.Baz can be lost.

On the other hand, when specifying the resourceVersion, one of the PUTs will fail, since whichever write succeeds changes the resourceVersion for Foo.

resourceVersion may be used as a precondition for other operations (e.g., GET, DELETE) in the future, such as for read-after-write consistency in the presence of caching.

"Watch" operations specify resourceVersion using a query parameter. It is used to specify the point at which to begin watching the specified resources. This may be used to ensure that no mutations are missed between a GET of a resource (or list of resources) and a subsequent Watch, even if the current version of the resource is more recent. This is currently the main reason that list operations (GET on a collection) return resourceVersion.


## Serialization Format

APIs may return alternative representations of any resource in response to an Accept header or under alternative endpoints, but the default serialization for input and output of API responses MUST be JSON.

All dates should be serialized as RFC3339 strings.


## Units

Units must either be explicit in the field name (e.g., `timeoutSeconds`), or must be specified as part of the value (e.g., `resource.Quantity`). Which approach is preferred is TBD.


## Selecting Fields

Some APIs may need to identify which field in a JSON object is invalid, or to reference a value to extract from a separate resource. The current recommendation is to use standard JavaScript syntax for accessing that field, assuming the JSON object was transformed into a JavaScript object.

Examples:

* Find the field "current" in the object "state" in the second item in the array "fields": `fields[0].state.current`

TODO: Plugins, extensions, nested kinds, headers


## HTTP Status codes

The server will respond with HTTP status codes that match the HTTP spec. See the section below for a breakdown of the types of status codes the server will send.

The following HTTP status codes may be returned by the API.

#### Success codes

* `200 StatusOK`
  * Indicates that the request completed successfully.
* `201 StatusCreated`
  * Indicates that the request to create kind completed successfully.
* `204 StatusNoContent`
  * Indicates that the request completed successfully, and the response contains no body.
  * Returned in response to HTTP OPTIONS requests.

#### Error codes

* `307 StatusTemporaryRedirect`
  * Indicates that the address for the requested resource has changed.
  * Suggested client recovery behavior
    * Follow the redirect.
* `400 StatusBadRequest`
  * Indicates the requested is invalid.
  * Suggested client recovery behavior:
    * Do not retry. Fix the request.
* `401 StatusUnauthorized`
  * Indicates that the server can be reached and understood the request, but refuses to take any further action, because the client must provide authorization. If the client has provided authorization, the server is indicating the provided authorization is unsuitable or invalid.
  * Suggested client recovery behavior
    * If the user has not supplied authorization information, prompt them for the appropriate credentials
    * If the user has supplied authorization information, inform them their credentials were rejected and optionally prompt them again.
* `403 StatusForbidden`
  * Indicates that the server can be reached and understood the request, but refuses to take any further action, because it is configured to deny access for some reason to the requested resource by the client.
  * Suggested client recovery behavior
    * Do not retry. Fix the request.
* `404 StatusNotFound`
  * Indicates that the requested resource does not exist.
  * Suggested client recovery behavior
    * Do not retry. Fix the request.
* `405 StatusMethodNotAllowed`
  * Indicates that the action the client attempted to perform on the resource was not supported by the code.
  * Suggested client recovery behavior
    * Do not retry. Fix the request.
* `409 StatusConflict`
  * Indicates that either the resource the client attempted to create already exists or the requested update operation cannot be completed due to a conflict.
  * Suggested client recovery behavior
  * * If creating a new resource
  *   * Either change the identifier and try again, or GET and compare the fields in the pre-existing object and issue a PUT/update to modify the existing object.
  * * If updating an existing resource:
      * See `Conflict` from the `status` response section below on how to retrieve more information about the nature of the conflict.
      * GET and compare the fields in the pre-existing object, merge changes (if still valid according to preconditions), and retry with the updated request (including `ResourceVersion`).
* `422 StatusUnprocessableEntity`
  * Indicates that the requested create or update operation cannot be completed due to invalid data provided as part of the request.
  * Suggested client recovery behavior
    * Do not retry. Fix the request.
* `429 StatusTooManyRequests`
  * Indicates that the either the client rate limit has been exceeded or the server has received more requests then it can process.
  * Suggested client recovery behavior:
    * Read the `Retry-After` HTTP header from the response, and wait at least that long before retrying.
* `500 StatusInternalServerError`
  * Indicates that the server can be reached and understood the request, but either an unexpected internal error occurred and the outcome of the call is unknown, or the server cannot complete the action in a reasonable time (this maybe due to temporary server load or a transient communication issue with another server).
  * Suggested client recovery behavior:
    * Retry with exponential backoff.
* `503 StatusServiceUnavailable`
  * Indicates that required service is unavailable.
  * Suggested client recovery behavior:
    * Retry with exponential backoff.
* `504 StatusServerTimeout`
  * Indicates that the request could not be completed within the given time. Clients can get this response ONLY when they specified a timeout param in the request.
  * Suggested client recovery behavior:
    * Increase the value of the timeout param and retry with exponential backoff

## Response Status Kind

Kubernetes will always return the `Status` kind from any API endpoint when an error occurs.
Clients SHOULD handle these types of objects when appropriate.

A `Status` kind will be returned by the API in two cases:
  * When an operation is not successful (i.e. when the server would return a non 2xx HTTP status code).
  * When a HTTP `DELETE` call is successful.

The status object is encoded as JSON and provided as the body of the response.  The status object contains fields for humans and machine consumers of the API to get more detailed information for the cause of the failure. The information in the status object supplements, but does not override, the HTTP status code's meaning. When fields in the status object have the same meaning as generally defined HTTP headers and that header is returned with the response, the header should be considered as having higher priority.

**Example:**

```console
$ curl -v -k -H "Authorization: Bearer WhCDvq4VPpYhrcfmF6ei7V9qlbqTubUc" https://10.240.122.184:443/api/v1/namespaces/default/pods/grafana

> GET /api/v1/namespaces/default/pods/grafana HTTP/1.1
> User-Agent: curl/7.26.0
> Host: 10.240.122.184
> Accept: */*
> Authorization: Bearer WhCDvq4VPpYhrcfmF6ei7V9qlbqTubUc
> 

< HTTP/1.1 404 Not Found
< Content-Type: application/json
< Date: Wed, 20 May 2015 18:10:42 GMT
< Content-Length: 232
< 
{
  "kind": "Status",
  "apiVersion": "v1",
  "metadata": {},
  "status": "Failure",
  "message": "pods \"grafana\" not found",
  "reason": "NotFound",
  "details": {
    "name": "grafana",
    "kind": "pods"
  },
  "code": 404
}
```

`status` field contains one of two possible values:
* `Success`
* `Failure`

`message` may contain human-readable description of the error

`reason` may contain a machine-readable description of why this operation is in the `Failure` status. If this value is empty there is no information available. The `reason` clarifies an HTTP status code but does not override it.

`details` may contain extended data associated with the reason. Each reason may define its own extended details. This field is optional and the data returned is not guaranteed to conform to any schema except that defined by the reason type.

Possible values for the `reason` and `details` fields:
* `BadRequest`
  * Indicates that the request itself was invalid, because the request doesn't make any sense, for example deleting a read-only object.
  * This is different than `status reason` `Invalid` above which indicates that the API call could possibly succeed, but the data was invalid.
  * API calls that return BadRequest can never succeed.
  * Http status code: `400 StatusBadRequest`
* `Unauthorized`
  * Indicates that the server can be reached and understood the request, but refuses to take any further action without the client providing appropriate authorization. If the client has provided authorization, this error indicates the provided credentials are insufficient or invalid.
  * Details (optional):
    * `kind string`
      * The kind attribute of the unauthorized resource (on some operations may differ from the requested resource).
    * `name string`
      * The identifier of the unauthorized resource.
   * HTTP status code: `401 StatusUnauthorized`
* `Forbidden`
  * Indicates that the server can be reached and understood the request, but refuses to take any further action, because it is configured to deny access for some reason to the requested resource by the client.
  * Details (optional):
    * `kind string`
      * The kind attribute of the forbidden resource (on some operations may differ from the requested resource).
    * `name string`
      * The identifier of the forbidden resource.
	 * HTTP status code: `403 StatusForbidden`
* `NotFound`
  * Indicates that one or more resources required for this operation could not be found.
  * Details (optional):
    * `kind string`
      * The kind attribute of the missing resource (on some operations may differ from the requested resource).
    * `name string`
      * The identifier of the missing resource.
  * HTTP status code: `404 StatusNotFound`
* `AlreadyExists`
  * Indicates that the resource you are creating already exists.
  * Details (optional):
    * `kind string`
      * The kind attribute of the conflicting resource.
    * `name string`
      * The identifier of the conflicting resource.
  * HTTP status code: `409 StatusConflict`
* `Conflict`
  * Indicates that the requested update operation cannot be completed due to a conflict. The client may need to alter the request. Each resource may define custom details that indicate the nature of the conflict.
  * HTTP status code: `409 StatusConflict`
* `Invalid`
  * Indicates that the requested create or update operation cannot be completed due to invalid data provided as part of the request.
  * Details (optional):
    * `kind string`
      * the kind attribute of the invalid resource
    * `name string`
      * the identifier of the invalid resource
    * `causes`
      * One or more `StatusCause` entries indicating the data in the provided resource that was invalid. The `reason`, `message`, and `field` attributes will be set.
  * HTTP status code: `422 StatusUnprocessableEntity`
* `Timeout`
  * Indicates that the request could not be completed within the given time. Clients may receive this response if the server has decided to rate limit the client, or if the server is overloaded and cannot process the request at this time.
  * Http status code: `429 TooManyRequests`
  * The server should set the `Retry-After` HTTP header and return `retryAfterSeconds` in the details field of the object. A value of `0` is the default.
* `ServerTimeout`
  * Indicates that the server can be reached and understood the request, but cannot complete the action in a reasonable time. This maybe due to temporary server load or a transient communication issue with another server.
    * Details (optional):
      * `kind string`
        * The kind attribute of the resource being acted on.
      * `name string`
        * The operation that is being attempted.
  * The server should set the `Retry-After` HTTP header and return `retryAfterSeconds` in the details field of the object. A value of `0` is the default.
  * Http status code: `504 StatusServerTimeout`
* `MethodNotAllowed`
  * Indicates that the action the client attempted to perform on the resource was not supported by the code.
  * For instance, attempting to delete a resource that can only be created.
  * API calls that return MethodNotAllowed can never succeed.
  * Http status code: `405 StatusMethodNotAllowed`
* `InternalError`
  * Indicates that an internal error occurred, it is unexpected and the outcome of the call is unknown.
  * Details (optional):
    * `causes`
      * The original error.
  * Http status code: `500 StatusInternalServerError`

`code` may contain the suggested HTTP return code for this status.


## Events

TODO: Document events (refer to another doc for details)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/api-conventions.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
