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

Updated: 10/8/2015

*This document is oriented at users who want a deeper understanding of the Kubernetes
API structure, and developers wanting to extend the Kubernetes API.  An introduction to
using resources with kubectl can be found in [Working with resources](../user-guide/working-with-resources.md).*

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
  - [Object references](#object-references)
  - [HTTP Status codes](#http-status-codes)
      - [Success codes](#success-codes)
      - [Error codes](#error-codes)
  - [Response Status Kind](#response-status-kind)
  - [Events](#events)
  - [Naming conventions](#naming-conventions)
  - [Label, selector, and annotation conventions](#label-selector-and-annotation-conventions)

<!-- END MUNGE: GENERATED_TOC -->

The conventions of the [Kubernetes API](../api.md) (and related APIs in the ecosystem) are intended to ease client development and ensure that configuration mechanisms can be implemented that work across a diverse set of use cases consistently.

The general style of the Kubernetes API is RESTful - clients create, update, delete, or retrieve a description of an object via the standard HTTP verbs (POST, PUT, DELETE, and GET) - and those APIs preferentially accept and return JSON. Kubernetes also exposes additional endpoints for non-standard verbs and allows alternative content types. All of the JSON accepted and returned by the server has a schema, identified by the "kind" and "apiVersion" fields. Where relevant HTTP header fields exist, they should mirror the content of JSON fields, but the information should not be represented only in the HTTP header.

The following terms are defined:

* **Kind** the name of a particular object schema (e.g. the "Cat" and "Dog" kinds would have different attributes and properties)
* **Resource** a representation of a system entity, sent or retrieved as JSON via HTTP to the server. Resources are exposed via:
  * Collections - a list of resources of the same type, which may be queryable
  * Elements - an individual resource, addressable via a URL

Each resource typically accepts and returns data of a single kind.  A kind may be accepted or returned by multiple resources that reflect specific use cases. For instance, the kind "Pod" is exposed as a "pods" resource that allows end users to create, update, and delete pods, while a separate "pod status" resource (that acts on "Pod" kind) allows automated processes to update a subset of the fields in that resource.

Resource collections should be all lowercase and plural, whereas kinds are CamelCase and singular.


## Types (Kinds)

Kinds are grouped into three categories:

1. **Objects** represent a persistent entity in the system.

   Creating an API object is a record of intent - once created, the system will work to ensure that resource exists. All API objects have common metadata.

   An object may have multiple resources that clients can use to perform specific actions that create, update, delete, or get.

   Examples: `Pod`, `ReplicationController`, `Service`, `Namespace`, `Node`.

2. **Lists** are collections of **resources** of one (usually) or more (occasionally) kinds.

   Lists have a limited set of common metadata. All lists use the "items" field to contain the array of objects they return.

   Most objects defined in the system should have an endpoint that returns the full set of resources, as well as zero or more endpoints that return subsets of the full list. Some objects may be singletons (the current user, the system defaults) and may not have lists.

   In addition, all lists that return objects with labels should support label filtering (see [docs/user-guide/labels.md](../user-guide/labels.md), and most lists should support filtering by fields.

   Examples: PodLists, ServiceLists, NodeLists

   TODO: Describe field filtering below or in a separate doc.

3. **Simple** kinds are used for specific actions on objects and for non-persistent entities.

   Given their limited scope, they have the same set of limited common metadata as lists.

   For instance, the "Status" kind is returned when errors occur and is not persisted in the system.

   Many simple resources are "subresources", which are rooted at API paths of specific resources. When resources wish to expose alternative actions or views that are closely coupled to a single resource, they should do so using new sub-resources. Common subresources include:

   * `/binding`: Used to bind a resource representing a user request (e.g., Pod, PersistentVolumeClaim) to a cluster infrastructure resource (e.g., Node, PersistentVolume).
   * `/status`: Used to write just the status portion of a resource. For example, the `/pods` endpoint only allows updates to `metadata` and `spec`, since those reflect end-user intent. An automated process should be able to modify status for users to see by sending an updated Pod kind to the server to the "/pods/&lt;name&gt;/status" endpoint - the alternate endpoint allows different rules to be applied to the update, and access to be appropriately restricted.
   * `/scale`: Used to read and write the count of a resource in a manner that is independent of the specific resource schema.

   Two additional subresources, `proxy` and `portforward`, provide access to cluster resources as described in [docs/user-guide/accessing-the-cluster.md](../user-guide/accessing-the-cluster.md).

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
* generation: a sequence number representing a specific generation of the desired state. Set by the system and monotonically increasing, per-resource. May be compared, such as for RAW and WAW consistency.
* creationTimestamp: a string representing an RFC 3339 date of the date and time an object was created
* deletionTimestamp: a string representing an RFC 3339 date of the date and time after which this resource will be deleted. This field is set by the server when a graceful deletion is requested by the user, and is not directly settable by a client. The resource will be deleted (no longer visible from resource lists, and not reachable by name) after the time in this field. Once set, this value may not be unset or be set further into the future, although it may be shortened or the resource may be deleted prior to this time.
* labels: a map of string keys and values that can be used to organize and categorize objects (see [docs/user-guide/labels.md](../user-guide/labels.md))
* annotations: a map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about this object (see [docs/user-guide/annotations.md](../user-guide/annotations.md))

Labels are intended for organizational purposes by end users (select the pods that match this label query). Annotations enable third-party automation and tooling to decorate objects with additional metadata for their own use.

#### Spec and Status

By convention, the Kubernetes API makes a distinction between the specification of the desired state of an object (a nested object field called "spec") and the status of the object at the current time (a nested object field called "status"). The specification is a complete description of the desired state, including configuration settings provided by the user, [default values](#defaulting) expanded by the system, and properties initialized or otherwise changed after creation by other ecosystem components (e.g., schedulers, auto-scalers), and is persisted in stable storage with the API object. If the specification is deleted, the object will be purged from the system. The status summarizes the current state of the object in the system, and is usually persisted with the object by an automated processes but may be generated on the fly. At some cost and perhaps some temporary degradation in behavior, the status could be reconstructed by observation if it were lost.

When a new version of an object is POSTed or PUT, the "spec" is updated and available immediately. Over time the system will work to bring the "status" into line with the "spec". The system will drive toward the most recent "spec" regardless of previous versions of that stanza. In other words, if a value is changed from 2 to 5 in one PUT and then back down to 3 in another PUT the system is not required to 'touch base' at 5 before changing the "status" to 3. In other words, the system's behavior is *level-based* rather than *edge-based*. This enables robust behavior in the presence of missed intermediate state changes.

The Kubernetes API also serves as the foundation for the declarative configuration schema for the system. In order to facilitate level-based operation and expression of declarative configuration, fields in the specification should have declarative rather than imperative names and semantics -- they represent the desired state, not actions intended to yield the desired state.

The PUT and POST verbs on objects MUST ignore the "status" values, to avoid accidentally overwriting the status in read-modify-write scenarios. A `/status` subresource MUST be provided to enable system components to update statuses of resources they manage.

Otherwise, PUT expects the whole object to be specified. Therefore, if a field is omitted it is assumed that the client wants to clear that field's value. The PUT verb does not accept partial updates. Modification of just part of an object may be achieved by GETting the resource, modifying part of the spec, labels, or annotations, and then PUTting it back. See [concurrency control](#concurrency-control-and-consistency), below, regarding read-modify-write consistency when using this pattern. Some objects may expose alternative resource representations that allow mutation of the status, or performing custom actions on the object.

All objects that represent a physical resource whose state may vary from the user's desired intent SHOULD have a "spec" and a "status".  Objects whose state cannot vary from the user's desired intent MAY have only "spec", and MAY rename "spec" to a more appropriate name.

Objects that contain both spec and status should not contain additional top-level fields other than the standard metadata fields.

##### Typical status properties

**Conditions** represent the latest available observations of an object's current state. Objects may report multiple conditions, and new types of conditions may be added in the future. Therefore, conditions are represented using a list/slice, where all have similar structure.

The `FooCondition` type for some resource type `Foo` may include a subset of the following fields, but must contain at least `type` and `status` fields:

```golang
	Type               FooConditionType  `json:"type" description:"type of Foo condition"`
	Status             ConditionStatus   `json:"status" description:"status of the condition, one of True, False, Unknown"`
	LastHeartbeatTime  unversioned.Time         `json:"lastHeartbeatTime,omitempty" description:"last time we got an update on a given condition"`
	LastTransitionTime unversioned.Time         `json:"lastTransitionTime,omitempty" description:"last time the condition transit from one status to another"`
	Reason             string            `json:"reason,omitempty" description:"one-word CamelCase reason for the condition's last transition"`
	Message            string            `json:"message,omitempty" description:"human-readable message indicating details about last transition"`
```

Additional fields may be added in the future.

Conditions should be added to explicitly convey properties that users and components care about rather than requiring those properties to be inferred from other observations.

Condition status values may be `True`, `False`, or `Unknown`. The absence of a condition should be interpreted the same as `Unknown`.

In general, condition values may change back and forth, but some condition transitions may be monotonic, depending on the resource and condition type. However, conditions are observations and not, themselves, state machines, nor do we define comprehensive state machines for objects, nor behaviors associated with state transitions. The system is level-based rather than edge-triggered, and should assume an Open World.

A typical oscillating condition type is `Ready`, which indicates the object was believed to be fully operational at the time it was last probed. A possible monotonic condition could be `Succeeded`. A `False` status for `Succeeded` would imply failure. An object that was still active would not have a `Succeeded` condition, or its status would be `Unknown`.

Some resources in the v1 API contain fields called **`phase`**, and associated `message`, `reason`, and other status fields. The pattern of using `phase` is deprecated. Newer API types should use conditions instead. Phase was essentially a state-machine enumeration field, that contradicted [system-design principles](../design/principles.md#control-logic) and hampered evolution, since [adding new enum values breaks backward compatibility](api_changes.md). Rather than encouraging clients to infer implicit properties from phases, we intend to explicitly expose the conditions that clients need to monitor. Conditions also have the benefit that it is possible to create some conditions with uniform meaning across all resource types, while still exposing others that are unique to specific resource types. See [#7856](http://issues.k8s.io/7856) for more details and discussion.

In condition types, and everywhere else they appear in the API, **`Reason`** is intended to be a one-word, CamelCase representation of the category of cause of the current status, and **`Message`** is intended to be a human-readable phrase or sentence, which may contain specific details of the individual occurrence. `Reason` is intended to be used in concise output, such as one-line `kubectl get` output, and in summarizing occurrences of causes, whereas `Message` is intended to be presented to users in detailed status explanations, such as `kubectl describe` output.

Historical information status (e.g., last transition time, failure counts) is only provided with reasonable effort, and is not guaranteed to not be lost.

Status information that may be large (especially proportional in size to collections of other resources, such as lists of references to other objects -- see below) and/or rapidly changing, such as [resource usage](../design/resources.md#usage-data), should be put into separate objects, with possibly a reference from the original object. This helps to ensure that GETs and watch remain reasonably efficient for the majority of clients, which may not need that data.

Some resources report the `observedGeneration`, which is the `generation` most recently observed by the component responsible for acting upon changes to the desired state of the resource. This can be used, for instance, to ensure that the reported status reflects the most recent desired status.

#### References to related objects

References to loosely coupled sets of objects, such as [pods](../user-guide/pods.md) overseen by a [replication controller](../user-guide/replication-controller.md), are usually best referred to using a [label selector](../user-guide/labels.md). In order to ensure that GETs of individual objects remain bounded in time and space, these sets may be queried via separate API queries, but will not be expanded in the referring object's status.

References to specific objects, especially specific resource versions and/or specific fields of those objects, are specified using the `ObjectReference` type (or other types representing strict subsets of it). Unlike partial URLs, the ObjectReference type facilitates flexible defaulting of fields from the referring object or other contextual information.

References in the status of the referee to the referrer may be permitted, when the references are one-to-one and do not need to be frequently updated, particularly in an edge-based manner.

#### Lists of named subobjects preferred over maps

Discussed in [#2004](http://issue.k8s.io/2004) and elsewhere. There are no maps of subobjects in any API objects. Instead, the convention is to use a list of subobjects containing name fields.

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

This rule maintains the invariant that all JSON/YAML keys are fields in API objects. The only exceptions are pure maps in the API (currently, labels, selectors, annotations, data), as opposed to sets of subobjects.

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
* GET /&lt;resourceNamePlural&gt;&amp;watch=true - Receive a stream of JSON objects corresponding to changes made to any resource of the given kind over time.

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

Units must either be explicit in the field name (e.g., `timeoutSeconds`), or must be specified as part of the value (e.g., `resource.Quantity`). Which approach is preferred is TBD, though currently we use the `fooSeconds` convention for durations.


## Selecting Fields

Some APIs may need to identify which field in a JSON object is invalid, or to reference a value to extract from a separate resource. The current recommendation is to use standard JavaScript syntax for accessing that field, assuming the JSON object was transformed into a JavaScript object, without the leading dot, such as `metadata.name`.

Examples:

* Find the field "current" in the object "state" in the second item in the array "fields": `fields[1].state.current`

## Object references

Object references should either be called `fooName` if referring to an object of kind `Foo` by just the name (within the current namespace, if a namespaced resource), or should be called `fooRef`, and should contain a subset of the fields of the `ObjectReference` type.


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

`reason` may contain a machine-readable, one-word, CamelCase description of why this operation is in the `Failure` status. If this value is empty there is no information available. The `reason` clarifies an HTTP status code but does not override it.

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

Events are complementary to status information, since they can provide some historical information about status and occurrences in addition to current or previous status. Generate events for situations users or administrators should be alerted about.

Choose a unique, specific, short, CamelCase reason for each event category. For example, `FreeDiskSpaceInvalid` is a good event reason because it is likely to refer to just one situation, but `Started` is not a good reason because it doesn't sufficiently indicate what started, even when combined with other event fields.

`Error creating foo` or `Error creating foo %s` would be appropriate for an event message, with the latter being preferable, since it is more informational.

Accumulate repeated events in the client, especially for frequent events, to reduce data volume, load on the system, and noise exposed to users.

## Naming conventions

* Go field names must be CamelCase. JSON field names must be camelCase. Other than capitalization of the initial letter, the two should almost always match. No underscores nor dashes in either.
* Field and resource names should be declarative, not imperative (DoSomething, SomethingDoer, DoneBy, DoneAt).
* `Minion` has been deprecated in favor of `Node`. Use `Node` where referring to the node resource in the context of the cluster. Use `Host` where referring to properties of the individual physical/virtual system, such as `hostname`, `hostPath`, `hostNetwork`, etc.
* `FooController` is a deprecated kind naming convention. Name the kind after the thing being controlled instead (e.g., `Job` rather than `JobController`).
* The name of a field that specifies the time at which `something` occurs should be called `somethingTime`. Do not use `stamp` (e.g., `creationTimestamp`).
* We use the `fooSeconds` convention for durations, as discussed in the [units subsection](#units).
  * `fooPeriodSeconds` is preferred for periodic intervals and other waiting periods (e.g., over `fooIntervalSeconds`).
  * `fooTimeoutSeconds` is preferred for inactivity/unresponsiveness deadlines.
  * `fooDeadlineSeconds` is preferred for activity completion deadlines.
* Do not use abbreviations in the API, except where they are extremely commonly used, such as "id", "args", or "stdin".
* Acronyms should similarly only be used when extremely commonly known. All letters in the acronym should have the same case, using the appropriate case for the situation. For example, at the beginning of a field name, the acronym should be all lowercase, such as "httpGet". Where used as a constant, all letters should be uppercase, such as "TCP" or "UDP".
* The name of a field referring to another resource of kind `Foo` by name should be called `fooName`. The name of a field referring to another resource of kind `Foo` by ObjectReference (or subset thereof) should be called `fooRef`.
* More generally, include the units and/or type in the field name if they could be ambiguous and they are not specified by the value or value type.

## Label, selector, and annotation conventions

Labels are the domain of users. They are intended to facilitate organization and management of API resources using attributes that are meaningful to users, as opposed to meaningful to the system. Think of them as user-created mp3 or email inbox labels, as opposed to the directory structure used by a program to store its data. The former is enables the user to apply an arbitrary ontology, whereas the latter is implementation-centric and inflexible. Users will use labels to select resources to operate on, display label values in CLI/UI columns, etc. Users should always retain full power and flexibility over the label schemas they apply to labels in their namespaces.

However, we should support conveniences for common cases by default. For example, what we now do in ReplicationController is automatically set the RC's selector and labels to the labels in the pod template by default, if they are not already set. That ensures that the selector will match the template, and that the RC can be managed using the same labels as the pods it creates. Note that once we generalize selectors, it won't necessarily be possible to unambiguously generate labels that match an arbitrary selector.

If the user wants to apply additional labels to the pods that it doesn't select upon, such as to facilitate adoption of pods or in the expectation that some label values will change, they can set the selector to a subset of the pod labels. Similarly, the RC's labels could be initialized to a subset of the pod template's labels, or could include additional/different labels.

For disciplined users managing resources within their own namespaces, it's not that hard to consistently apply schemas that ensure uniqueness. One just needs to ensure that at least one value of some label key in common differs compared to all other comparable resources. We could/should provide a verification tool to check that. However, development of conventions similar to the examples in [Labels](../user-guide/labels.md) make uniqueness straightforward. Furthermore, relatively narrowly used namespaces (e.g., per environment, per application) can be used to reduce the set of resources that could potentially cause overlap.

In cases where users could be running misc. examples with inconsistent schemas, or where tooling or components need to programmatically generate new objects to be selected, there needs to be a straightforward way to generate unique label sets. A simple way to ensure uniqueness of the set is to ensure uniqueness of a single label value, such as by using a resource name, uid, resource hash, or generation number.

Problems with uids and hashes, however, include that they have no semantic meaning to the user, are not memorable nor readily recognizable, and are not predictable. Lack of predictability obstructs use cases such as creation of a replication controller from a pod, such as people want to do when exploring the system, bootstrapping a self-hosted cluster, or deletion and re-creation of a new RC that adopts the pods of the previous one, such as to rename it. Generation numbers are more predictable and much clearer, assuming there is a logical sequence. Fortunately, for deployments that's the case. For jobs, use of creation timestamps is common internally. Users should always be able to turn off auto-generation, in order to permit some of the scenarios described above. Note that auto-generated labels will also become one more field that needs to be stripped out when cloning a resource, within a namespace, in a new namespace, in a new cluster, etc., and will need to be ignored around when updating a resource via patch or read-modify-write sequence.

Inclusion of a system prefix in a label key is fairly hostile to UX. A prefix is only necessary in the case that the user cannot choose the label key, in order to avoid collisions with user-defined labels. However, I firmly believe that the user should always be allowed to select the label keys to use on their resources, so it should always be possible to override default label keys.

Therefore, resources supporting auto-generation of unique labels should have a `uniqueLabelKey` field, so that the user could specify the key if they wanted to, but if unspecified, it could be set by default, such as to the resource type, like job, deployment, or replicationController. The value would need to be at least spatially unique, and perhaps temporally unique in the case of job.

Annotations have very different intended usage from labels. We expect them to be primarily generated and consumed by tooling and system extensions. I'm inclined to generalize annotations to permit them to directly store arbitrary json. Rigid names and name prefixes make sense, since they are analogous to API fields.

In fact, experimental API fields, including those used to represent fields of newer alpha/beta API versions in the older stable storage version, may be represented as annotations with the form `something.experimental.kubernetes.io/name`.  For example `net.experimental.kubernetes.io/policy` might represent an experimental network policy field.

Other advice regarding use of labels, annotations, and other generic map keys by Kubernetes components and tools:
  - Key names should be all lowercase, with words separated by dashes, such as `desired-replicas`
  - Prefix the key with `kubernetes.io/` or `foo.kubernetes.io/`, preferably the latter if the label/annotation is specific to `foo`
    - For instance, prefer `service-account.kubernetes.io/name` over `kubernetes.io/service-account.name`
  - Use annotations to store API extensions that the controller responsible for the resource doesn't need to know about, experimental fields that aren't intended to be generally used API fields, etc. Beware that annotations aren't automatically handled by the API conversion machinery.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/api-conventions.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
