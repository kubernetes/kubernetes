API Conventions
===============

The conventions of the Kubernetes API (and related APIs in the ecosystem) are intended to ease client development and ensure that configuration mechanisms can be implemented that work across a diverse set of use cases consistently.

The general style of the Kubernetes API is RESTful - clients create, update, delete, or retrieve a description of an object via the standard HTTP verbs (POST, PUT, DELETE, and GET) - and those APIs preferentially accept and return JSON. Kubernetes also exposes additional endpoints for non-standard verbs and allows alternative content types. All of the JSON accepted and returned by the server has a schema, identified by the "kind" and "apiVersion" fields.

The following terms are defined:

* **Kind** the name of a particular object schema (e.g. the "Cat" and "Dog" kinds would have different attributes and properties)
* **Resource** a representation of a system entity, sent or retrieved as JSON via HTTP to the server. Resources are exposed via:
  * Collections - a list of resources of the same type, which may be queryable
  * Elements - an individual resource, addressible via a URL

Each resource typically accepts and returns data of a single kind.  A kind may be accepted or returned by multiple resources that reflect specific use cases. For instance, the kind "pod" is exposed as a "pods" resource that allows end users to create, update, and delete pods, while a separate "pod status" resource (that acts on "pod" kind) allows automated processes to update a subset of the fields in that resource. A "restart" resource might be exposed for a number of different resources to allow the same action to have different results for each object.


Types (Kinds)
-------------

Kinds are grouped into three categories:

1. **Objects** represent a persistent entity in the system.

   Creating an API object is a record of intent - once created, the system will work to ensure that resource exists. All API objects have common metadata.

   An object may have multiple resources that clients can use to perform specific actions than create, update, delete, or get.

   Examples: Pods, ReplicationControllers, Services, Namespaces, Nodes

2. **Lists** are collections of **resources** of one (usually) or more (occasionally) kinds.

   Lists have a limited set of common metadata. All lists use the "items" field to contain the array of objects they return.

   Most objects defined in the system should have an endpoint that returns the full set of resources, as well as zero or more endpoints that return subsets of the full list. Some objects may be singletons (the current user, the system defaults) and may not have lists.

   In addition, all lists that return objects with labels should support label filtering (see [labels.md](labels.md), and most lists should support filtering by fields.

   Examples: PodLists, ServiceLists, NodeLists

   TODO: Describe field filtering below or in a separate doc.

3. **Simple** kinds are used for specific actions on objects and for non-persistent entities.

   Given their limited scope, they have the same set of limited common metadata as lists.

   The "size" action may accept a simple resource that has only a single field as input (the number of things). The "status" kind is returned when errors occur and is not persisted in the system.

   Examples: Binding, Status

The standard REST verbs (defined below) MUST return singular JSON objects. Some API endpoints may deviate from the strict REST pattern and return resources that are not singular JSON objects, such as streams of JSON objects or unstructured text log data.


### Resources

All JSON objects returned by an API MUST have the following fields:

* kind: a string that identifies the schema this object should have
* apiVersion: a string that identifies the version of the schema the object should have


### Objects

#### Metadata

Every object kind MUST have the following metadata in a nested object field called "metadata":

* namespace: a namespace is a DNS compatible subdomain that objects are subdivided into. The default namespace is 'default'.  See [namespaces.md](namespaces.md) for more.
* name: a string that uniquely identifies this object within the current namespace (see [identifiers.md](identifiers.md)). This value is used in the path when retrieving an individual object.
* uid: a unique in time and space value (typically an RFC 4122 generated identifier, see [identifiers.md](identifiers.md)) used to distinguish between objects with the same name that have been deleted and recreated

Every object SHOULD have the following metadata in a nested object field called "metadata":

* resourceVersion: a string that identifies the internal version of this object that can be used by clients to determine when objects have changed. This value MUST be treated as opaque by clients and passed unmodified back to the server. Clients should not assume that the resource version has meaning across namespaces, different kinds of resources, or different servers. (see [concurrency control](#concurrency-control-and-consistency), below, for more details)
* creationTimestamp: a string representing an RFC 3339 date of the date and time an object was created
* labels: a map of string keys and values that can be used to organize and categorize objects (see [labels.md](labels.md))
* annotations: a map of string keys and values that can be used by external tooling to store and retrieve arbitrary metadata about this object (see [annotations.md](annotations.md))

Labels are intended for organizational purposes by end users (select the pods that match this label query). Annotations enable third party automation and tooling to decorate objects with additional metadata for their own use.

#### Spec and Status

By convention, the Kubernetes API makes a distinction between the specification of the desired state of an object (a nested object field called "spec") and the status of the object at the current time (a nested object field called "status"). The specification is persisted in stable storage with the API object and reflects user input. The status is summarizes the current state of the object in the system, and is usually persisted with the object by an automated processes (but may be created on the fly).

For example, a pod object has a "spec" object field that defines how the pod should be run. The pod also has a "status" object field that shows details about what is happening on the host that is running the containers in the pod (if available) and a summarized "phase" string that indicates where the pod is in its lifecycle.

When a new version of an object is POSTed or PUT, the "spec" is updated and available immediately. Over time the system will work to bring the "status" into line with the "spec". The system will drive toward the most recent "spec" regardless of previous versions of that stanza. In other words, if a value is changed from 2 to 5 in one PUT and then back down to 3 in another PUT the system is not required to 'touch base' at 5 before changing the "status" to 3.

The PUT and POST verbs on objects will ignore the "status" values. Otherwise, PUT expects the whole object to be specified. Therefore, if a field is omitted it is assumed that the client wants to clear that field's value.

The PUT verb does not accept partial updates. Modification of just part of an object may be achieved by GETting the resource, modifying part of the spec, labels, or annotations, and then PUTting it back. See [concurrency control](#concurrency-control-and-consistency), below, regarding read-modify-write consistency when using this pattern. Some objects may expose alternative resource representations that allow mutation of the status, or performing custom actions on the object.

All objects that represent a physical resource whose state may vary from the user's desired intent SHOULD have a "spec" and a "status".  Objects whose state cannot vary from the user's desired intent MAY have only "spec", and MAY rename "spec" to a more appropriate name.

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

Every simple kind returned by the server, and any simple kind sent to the server that must support idempotency or optimistic concurency should return this value.Since simple resources are often used as input alternate actions that modify objects, the resource version of the simple resource should correspond to the resource version of the object.


Differing Representations
-------------------------

An API may represent a single entity in different ways for different clients, or transform an object after certain transitions in the system occur. In these cases, one request object may have two representations available as different resources, or different kinds.

An example is a Service, which represents the intent of the user to group a set of pods with common behavior on common ports. When Kubernetes detects a pod matches the service selector, the IP address and port of the pod are added to an Endpoints resource for that Service. The Endpoints resource exists only if the Service exists, but exposes only the IPs and ports of the selected pods.  The full service is represented by two distinct resources - under the original Service resource the user created, as well as in the Endpoints resource.

As another example, a "pod status" resource may accept a PUT with the "pod" kind, with different rules about what fields may be changed.

Future versions of Kubernetes may allow alternative encodings of objects beyond JSON.


Verbs on Resources
------------------

API resources should use the traditional REST pattern:

* GET /&lt;resourceNamePlural&gt; - Retrieve a list of type &lt;resourceName&gt;, e.g. GET /pods returns a list of Pods.
* POST /&lt;resourceNamePlural&gt; - Create a new resource from the JSON object provided by the client.
* GET /&lt;resourceNamePlural&gt;/&lt;name&gt; - Retrieves a single resource with the given name, e.g. GET /pods/first returns a Pod named 'first'.
* DELETE /&lt;resourceNamePlural&gt;/&lt;name&gt;  - Delete the single resource with the given name.
* PUT /&lt;resourceNamePlural&gt;/&lt;name&gt; - Update or create the resource with the given name with the JSON object provided by the client.

Kubernetes by convention exposes additional verbs as new root endpoints with singular names. Examples:

* GET /watch/&lt;resourceNamePlural&gt; - Receive a stream of JSON objects corresponding to changes made to any resource of the given kind over time.
* GET /watch/&lt;resourceNamePlural&gt;/&lt;name&gt; - Receive a stream of JSON objects corresponding to changes made to the named resource of the given kind over time.

These are verbs which change the fundamental type of data returned (watch returns a stream of JSON instead of a single JSON object). Support of additional verbs is not required for all object types.

Two additional verbs `redirect` and `proxy` provide access to cluster resources as described in [accessing-the-cluster.md](accessing-the-cluster.md).

When resources wish to expose alternative actions that are closely coupled to a single resource, they should do so using new sub-resources. An example is allowing automated processes to update the "status" field of a Pod. The `/pods` endpoint only allows updates to "metadata" and "spec", since those reflect end-user intent. An automated process should be able to modify status for users to see by sending an updated Pod kind to the server to the "/pods/&lt;name&gt;/status" endpoint - the alternate endpoint allows different rules to be applied to the update, and access to be appropriately restricted. Likewise, some actions like "stop" or "resize" are best represented as REST sub-resources that are POSTed to.  The POST action may require a simple kind to be provided if the action requires parameters, or function without a request body.

TODO: more documentation of Watch


Idempotency
-----------

All compatible Kubernetes APIs MUST support "name idempotency" and respond with an HTTP status code 409 when a request is made to POST an object that has the same name as an existing object in the system. See [identifiers.md](identifiers.md) for details.

TODO: name generation

Defaulting
----------

Default resource values are API version-specific, and they are applied during
the conversion from API-versioned declarative configuration to internal objects
representing the desired state (`Spec`) of the resource.

Incorporating the default values into the `Spec` ensures that `Spec` depicts the
full desired state so that it is easier for the system to determine how to
achieve the state, and for the user to know what to anticipate.


Concurrency Control and Consistency
-----------------------------------

Kubernetes leverages the concept of *resource versions* to achieve optimistic concurrency. All kubernetes resources have a "resourceVersion" field as part of their metadata. This resourceVersion is a string that identifies the internal version of an object that can be used by clients to determine when objects have changed. When a record is about to be updated, it's version is checked against a pre-saved value, and if it doesn't match, the update fails with a StatusConflict (HTTP status code 409).

The resourceVersion is changed by the server every time an object is modified. If resourceVersion is included with the PUT operation the system will verify that there have not been other successful mutations to the resource during a read/modify/write cycle, by verifying that the current value of resourceVersion matches the specified value.

The resourceVersion is currently backed by [etcd's modifiedIndex](https://coreos.com/docs/distributed-configuration/etcd-api/). However, it's important to note that the application should *not* rely on the implementation details of the versioning system maintained by kubernetes. We may change the implementation of resourceVersion in the future, such as to change it to a timestamp or per-object counter.

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


Serialization Format
--------------------

APIs may return alternative representations of any resource in response to an Accept header or under alternative endpoints, but the default serialization for input and output of API responses MUST be JSON.

All dates should be serialized as RFC3339 strings.


Selecting Fields
----------------

Some APIs may need to identify which field in a JSON object is invalid, or to reference a value to extract from a separate resource. The current recommendation is to use standard JavaScript syntax for accessing that field, assuming the JSON object was transformed into a JavaScript object.

Examples:

* Find the field "current" in the object "state" in the second item in the array "fields": `fields[0].state.current`

TODO: Plugins, extensions, nested kinds, headers


HTTP Status codes
-----------------

The following HTTP status codes may be returned by the API.

#### Success codes

* `200 StatusOK`
  * Indicates that the request completed succesfully.
* `201 StatusCreated`
  * Indicates that the request to create kind completed succesfully.
* `204 StatusNoContent`
  * Indicates that the request completed succesfully, and the response contains no body.
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
* `403 StatusForbidden`
  * Indicates that the server can be reached and understood the request, but refuses to take any further action, because it is configured to deny access for some reason to the requested resource by the client.
  * Suggested client recovery behavior
    * Do not retry. Fix the request.
* `404 StatusNotFound`
  * Indicates that the requested resource does not exist.
  * Suggested client recovery behavior
    * Do not retry. Fix the request.
* `405 StatusMethodNotAllowed`
  * Indicates that that the action the client attempted to perform on the resource was not supported by the code.
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
  * Indicates that the either the client rate limit has been exceeded or the server has recieved more requests then it can process.
  * Suggested client recovery behavior:
    * Read the ```Retry-After``` HTTP header from the response, and wait at least that long before retrying.
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

Response Status Kind
---------------------

Kubernetes MAY return the ```Status``` kind from any API endpoint. Clients SHOULD handle these types of objects when appropriate.

A ```Status``` kind MAY be returned by an API when an operation is successful (i.e. when an HTTP 200 status code is returned). In particular, delete APIs return the ```Status``` kind. The success status object simply contains a ```Status``` field set to ```Successs```.

A ```Status``` kind SHOULD be returned by an API when an operation is not successful (i.e. when the server would return a non 2xx HTTP status code). The status object contains fields for humans and machine consumers of the API to get more detailed information for the cause of the failure. The information in the status object supplements, but does not override, the HTTP status code's meaning.

**Example:**
```JSON
>HTTP Requst:
POST /api/v1beta1/events/ HTTP/1.1
Authorization: Basic ...

{empty body}

>HTTP Response:
HTTP/1.1 500 Internal Server Error
Server: nginx/1.2.1
Content-Type: application/json
Content-Length: 144

{
  "kind": "Status",
  "creationTimestamp": null,
  "apiVersion": "v1beta1",
  "status": "Failure",
  "message": "empty input",
  "code": 500
}
```

```status``` field contains one of two possible values:
* `Success`
* `Failure`

`message` may contain human-readable description of the error

```reason``` may contain a machine-readable description of why this operation is in the `Failure` status. If this value is empty there is no information available. The `reason` clarifies an HTTP status code but does not override it.

```details``` may contain extended data associated with the reason. Each reason may define its own extended details. This field is optional and the data returned is not guaranteed to conform to any schema except that defined by the reason type.

Possible values for the ```reason``` and ```details``` fields:
* `Forbidden`
  * Indicates that the server can be reached and understood the request, but refuses to take any further action, because it is configured to deny access for some reason to the requested resource by the client.
  * Details (optional):
    * `kind string`
      * The kind attribute of the forbidden resource (on some operations may differ from the requested resource).
    * `id   string`
      * The identifier of the forbidden resource.
	 * HTTP status code: `403 StatusForbidden`
* `NotFound`
  * Indicates that one or more resources required for this operation could not be found.
  * Details (optional):
    * `kind string`
      * The kind attribute of the missing resource (on some operations may differ from the requested resource).
    * `id   string`
      * The identifier of the missing resource.
  * HTTP status code: `404 StatusNotFound`
* `AlreadyExists`
  * Indicates that the resource you are creating already exists.
  * Details (optional):
    * `kind string`
      * The kind attribute of the conflicting resource.
    * `id   string`
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
    * `id   string`
      * the identifier of the invalid resource
    * `causes`
      * One or more `StatusCause` entries indicating the data in the provided resource that was invalid. The `reason`, `message`, and `field` attributes will be set.
  * HTTP status code: `422 StatusUnprocessableEntity`
* `ServerTimeout`
  * Indicates that the server can be reached and understood the request, but cannot complete the action in a reasonable time. This maybe due to temporary server load or a transient communication issue with another server.
    * Details (optional):
      * `kind string`
        * The kind attribute of the resource being acted on.
      * `id   string`
        * The operation that is being attempted.
  * Http status code: `500 StatusInternalServerError`
* `Timeout`
  * Indicates that the request could not be completed within the given time. Clients can get this response ONLY when they specified a timeout param in the request. The request might succeed with an increased value of timeout param.
  * Http status code: `504 StatusServerTimeout`
* `BadRequest`
  * Indicates that the request itself was invalid, because the request doesn't make any sense, for example deleting a read-only object.
  * This is different than `status reason` `Invalid` above which indicates that the API call could possibly succeed, but the data was invalid.
  * API calls that return BadRequest can never succeed.
  * Http status code: `400 StatusBadRequest`
* `MethodNotAllowed`
  * Indicates that that the action the client attempted to perform on the resource was not supported by the code.
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


Events
------

TODO: Document events (refer to another doc for details)


API Documentation
-----------------

API documentation can be found at [http://kubernetes.io/third_party/swagger-ui/](http://kubernetes.io/third_party/swagger-ui/).


