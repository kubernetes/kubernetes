<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.4/docs/proposals/alternate-api-representations.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Alternate representations of API resources

## Abstract

Naive clients benefit from allowing the server to returning resource information in a form
that is easy to represent or is more efficient when dealing with resources in bulk. It
should be possible to ask an API server to return a representation of one or more resources
in a way useful for:

* Retrieving a subset of object metadata in a list or watch of a resource, such as the
  metadata needed by the generic Garbage Collector or the Namespace Lifecycle Controller
* Dealing with generic operations like `Scale` correctly from a client across multiple API
  groups, versions, or servers
* Return a simple tabular representation of an object or list of objects for naive
  web or command-line clients to display (for `kubectl get`)
* Return a simple description of an object that can be displayed in a wide range of clients
  (for `kubectl describe`)
* Return the object with fields set by the server cleared (as `kubectl export`)

The server should allow a common mechanism for a client to request a resource be returned
in one of a number of possible forms.

Also, the server today contains a number of objects which are common across multiple groups,
but which clients must be able to deal with in a generic fashion. These objects - Status,
ListMeta, ObjectMeta, List, ListOptions, ExportOptions, and Scale - are embedded into each
group version but are actually part of a a shared API group. It must be possible for a naive
client to translate the Scale response returned by two different API group versions.


## Motivation

Currently it is difficult for a naive client (dealing only with the list of resources
presented by API discovery) to properly handle new and extended API groups, especially
as versions of those groups begin to evolve. It must be possible for a naive client to
perform a set of common operations across a wide range of groups and versions and leverage
a predictable schema.

We also foresee increasing difficulty in building clients that must deal with extensions -
there are at least 6 known web-ui or CLI implementations that need to display some
information about third party resources or additional API groups registered with a server
without requiring each of them to change.  Providing a server side implementation will
allow clietns to retrieve meaningful information for the `get` and `describe` style
operations even for new API groups.


## Implementation

The HTTP spec and the common REST paradigm provide mechanisms for clients to [negotiate
alternative representations of objects (RFC2616 14.1)](http://www.w3.org/Protocols/rfc2616/rfc2616.txt)
and for the server to correctly indicate a requested mechanism was chosen via the `Accept`
and `Content-Type` headers. This is a standard request response protocol intended to allow
clients to request the server choose a representation to return to the client based on the
server's capabilities. In RESTful terminology, a representation is simply a known schema that
the client is capable of handling - common schemas are HTML, JSON, XML, or protobuf, with the
possibility of the client and server further refining the requested output via either query
parameters or media type parameters.

In order to ensure that generic clients can properly deal with many different group versions,
we introduce the `api.k8s.io` group with version `v1` that grandfathers all existing resources
currently described as "unversioned".  A generic client may request that responses be applied
in this version. The contents of a particular API group version would continue to be bound into
other group versions (`status.v1.api.k8s.io` would be bound as `Status` into all existing
API groups).  We would remove the `unversioned` package and properly home these resources in
a real API group.


### Considerations around choosing an implementation

* We wish to avoid creating new resource *locations* (URLs) for existing resources
  * New resource locations complicate access control, caching, and proxying
  * We are still retrieving the same resource, just in an alternate representation,
    which matches our current use of the protobuf, JSON, and YAML serializations
  * We do not wish to alter the mechanism for authorization - a user with access
    to a particular resource in a given namespace should be limited regardless of
    the representation in use.
  * Allowing "all namespaces" to be listed would require us to create "fake" resources
    which would authorization
* We wish to support retrieving object representations in multiple schemas - JSON for
  simple clients and Protobuf for clients concerned with efficiency.
* Most clients will wish to retrieve the more efficient / simpler object, but for
  older servers will desire to fall back to the implict resource represented by
  the endpoint.
  * Over time, clients may need to request results in multiple API group versions
    because of breaking changes (when we introduce v2, clients that know v2 will want
    to ask for v2, then v1)
* We wish to preserve the greatest possible query parameter space for sub resources
  and special cases, which encourages us to avoid polluting the API with query
  parameters that can be otherwise represented.
* Because we expect not all extensions will implement protobuf, an efficient client
  must continue to be able to "fall-back" to JSON, such as for third party
  resources.
* We do not wish to create fake content-types like `application/json+kubernetes+v1+api.k8s.io`
  because the list of combinations is unbounded and our ability to encode specific values
  (like slashes) into the value is limited.

### Client negotiation of response representation

When a client wishes to request an alternate representation of an object, it should form
a valid `Accept` header containing one or more accepted representations, where each
representation is represented by a media-type and [media-type parameters](https://tools.ietf.org/html/rfc6838#section-4.3).
The server should omit representations that are unrecognized or in error - if no representations
are left after omission the server should return a `406 Not Acceptable` HTTP response.

The supported parameters are:

| Name | Value | Default | Description |
| ---- | ----- | ------- | ----------- |
| g | The group name of the desired response | Current group | The group the response is expected in. |
| v | The version of the desired response | Current version | The version the response is expected in. Note that this is separate from Group because `/` is not a valid character in Accept headers. |
| as | Kind name | None | If specified, transform the resource into the following kind (including the group and version parameters). |
| sv | The server group (`api.k8s.io`) version that should be applied to generic resources returned by this endpoint | Matching server version for the current group and version | If specified, the server should transform generic responses into this version of the server API group. |
| export | `1` | None | If specified, transform the resource prior to returning to omit defaulted fields. Additional arguments allowed in the query parameter. For legacy reasons, `?export=1` will continue to be supported on the request |
| pretty | `0`/`1` | `1` | If specified, apply formatting to the returned response that makes the serialization readable (for JSON, use indentation) |

For both export and the more complicated server side `kubectl get` cases, it's likely that
more parameters are required and should be specified as query parameters. However, the core
behavior is best represented as a variation on content-type. Supporting both is not limiting
in the short term as long as we can validate correctly.


### Example: Partial metadata retrieval

The client may request to the server to return the list of namespaces as a
`PartialObjectMetadata` kind, which is an object containing only `ObjectMeta` and
can be serialized as protobuf or JSON. This is expected to be significantly more
performant when controllers like the Garbage collector retrieve multiple objects.

    GET /api/v1/namespaces
    Accept: application/json;g=api.k8s.io,v=v1,as=PartialObjectMetadata, application/json

The server would respond with

    200 OK
    Content-Type: application/json;g=api.k8s.io,v=v1,as=PartialObjectMetadata
    {
      "apiVersion": "api.k8s.io/v1",
      "kind": "PartialObjectMetadataList",
      "items": [
        {
          "apiVersion": "api.k8s.io/v1",
          "kind": "PartialObjectMetadata",
          "metadata": {
            "name": "foo",
            "resourceVersion": "10",
            ...
          }
        },
        ...
      ]
    }

Note that the `as` parameter indicates to the server the Kind of the resource, but
the Kubernetes API convention of returning a List with a known schema continues. An older
server could ignore the presence of the `as` parameter on the media type and merely return
a `NamespaceList` and the client would either use the content-type or the object Kind
to distinguish. Because all responses are expected to be self-describing, an existing
Kubernetes client would be expected to differentiate on Kind.

An old server, not recognizing these parameters, would respond with:

    200 OK
    Content-Type: application/json
    {
      "apiVersion": "v1",
      "kind": "NamespaceList",
      "items": [
        {
          "apiVersion": "v1",
          "kind": "Namespace",
          "metadata": {
            "name": "foo",
            "resourceVersion": "10",
            ...
          }
        },
        ...
      ]
    }


### Example: Retrieving a known version of the Scale resource

Each API group that supports resources that can be scaled must expose a subresource on
their object that accepts GET or PUT with a `Scale` kind resource. This subresource acts
as a generic interface that a client that knows nothing about the underlying object can
use to modify the scale value of that resource. However, clients *must* be able to understand
the response the server provides, and over time the response may change and should therefore
be versioned. Our current API provides no way for a client to discover whether a `Scale`
response returned by `batch/v2alpha1` is the same as the `Scale` resource returned by
`autoscaling/v1`.

Under this proposal, to scale a generic resource a client would perform the following
operations:

    GET /api/v1/namespace/example/replicasets/test/scale
    Accept: application/json;g=api.k8s.io,v=v1,as=Scale, application/json

    200 OK
    Content-Type: application/json;g=api.k8s.io,v=v1,as=Scale
    {
      "apiVersion": "api.k8s.io/v1",
      "kind": "Scale",
      "spec": {
        "replicas": 1
      }
      ...
    }

The client, seeing that a generic response was returned (`api.k8s.io/v1`), knows that
the server supports accepting that resource as well, and performs a PUT:

    PUT /api/v1/namespace/example/replicasets/test/scale
    Accept: application/json;g=api.k8s.io,v=v1,as=Scale, application/json
    Content-Type: application/json
    {
      "apiVersion": "api.k8s.io/v1",
      "kind": "Scale",
      "spec": {
        "replicas": 2
      }
    }

    200 OK
    Content-Type: application/json;g=api.k8s.io,v=v1,as=Scale
    {
      "apiVersion": "api.k8s.io/v1",
      "kind": "Scale",
      "spec": {
        "replicas": 1
      }
      ...
    }

Note that the client still asks for the common Scale as the response so that it
can access the value it wants.


### Example: Retrieving an alternative representation of the resource for use in `kubectl get`

As new extension groups are added to the server, all clients must implement simple "view" logic
for each resource.  However, these views are specific to the resource in question, which only
the server is aware of. To make clients more tolerant of extension and third party resources,
it should be possible for clients to ask the server to present a resource or list of resources
in a tabular / descriptive format rather than raw JSON.

While the design of serverside tabular support is outside the scope of this proposal, a few
knows apply. The server must return a structured resource usable by both command line and
rich clients (web or IDE), which implies a schema, which implies JSON, and which means the
server should return a known Kind. For this example we will call that kind `TabularOutput`
to demonstrate the concept.

A server side resource would implement a transformation from their resource to `TabularOutput`
and the API machinery would translate a single item or a list of items (or a watch) into
the tabular resource.

A generic client wishing to display a tabular list for resources of type `v1.ReplicaSets` would
make the following call:

    GET /api/v1/namespaces/example/replicasets
    Accept: application/json;g=api.k8s.io,v=v1,as=TabularOutput, application/json

    200 OK
    Content-Type: application/json;g=api.k8s.io,v=v1,as=TabularOutput
    {
      "apiVersion": "api.k8s.io/v1",
      "kind": "TabularOutput",
      "columns": [
        {"name": "Name", "description": "The name of the resource"},
        {"name": "Resource Version", "description": "The version of the resource"},
        ...
      ],
      "items": [
        {"columns": ["name", "10", ...]},
        ...
      ]
    }

The client can then present that information as necessary. If the server returns the
resource list `v1.ReplicaSetList` the client knows that the server does not support tabular
output and so must fall back to a generic output form (perhaps using the existing
compiled in listers).

Note that `kubectl get` supports a number of parameters for modifying the response,
including whether to filter resources, whether to show a "wide" list, or whether to
turn certain labels into columns. Those options are best represented as query parameters
and transformed into a known type.


### Example: Versioning a ListOptions call to a generic API server

When retrieving lists of resources, the server transforms input query parameters like
`labels` and `fields` into a `ListOptions` type. It should be possible for a generic
client dealing with the server to be able to specify the version of ListOptions it
is sending to detect version skew.

Since this is an input and list is implemented with GET, it is not possible to send
a body and no Content-Type is possible.  For this approach, we recommend that the kind
and API version be specifiable via the GET call for further clarification:

New query parameters:

| Name | Value | Default | Description |
| ---- | ----- | ------- | ----------- |
| kind | The kind of parameters being sent | `ListOptions` (GET), `DeleteOptions` (DELETE) | The kind of the serialized struct, defaults to ListOptions on GET and DeleteOptions on DELETE. |
| apiVersion | The API version of the parameter struct | `api.k8s.io/v1` | May be altered to match the expected version.  Because we have not yet versioned ListOptions, this is safe to alter. |

To send ListOptions in the v2 future format, where the serialization of `resourceVersion`
is changed to `rv`, clients would provide:

    GET /api/v1/namespaces/example/replicasets?apiVersion=api.k8s.io/v2&rv=10

Before we introduce a second API group version, we would have to ensure old servers
properly reject apiVersions they do not understand.


### Impact on web infrastructure

In the past, web infrastructure and old browsers have coped poorly with the `Accept`
header.  However, most modern caching infrastructure properly supports `Vary: Accept`
and caching of responses has not been a significant requirement for Kubernetes APIs
to this point.


### Considerations for discoverability

To ensure clients can discover these endpoints, the Swagger and OpenAPI documents
should also include a set of example mime-types for each endpoint that are supported.
Specifically, the `produces` field on an individual operation can be used to list a
set of well known types. The description of the operation can include a stanza about
retrieving alternate representations.


## Alternatives considered

*  Implement only with query parameters

   To properly implement alternative resource versions must support multiple version
   support (ask for v2, then v1). The Accept mechanism already handles this sort of
   multi-version negotiation, while any approach based on query parameters would
   have to implement this option as well. In addition, some serializations may not
   be valid in all content types, so the client asking for TabularOutput in protobuf
   may also ask for TabularOutput in JSON - if TabularOutput is not valid in protobuf
   the server call fall back to JSON.

*  Use new resource paths - `/apis/autoscaling/v1/namespaces/example/horizontalpodautoscalermetadata`

   This leads to a proliferation of paths which will confuse automated tools and end
   users. Authorization, logging, audit may all need a way to map the two resources
   as equivalent, while clients would need a discovery mechanism that identifies a
   "same underlying object" relationship that is different from subresources.

*  Use a special HTTP header to denote the alternative representation

   Given the need to support multiple versions, this would be reimplementing Accept
   in a slightly different way, so we prefer to reuse Accept.


## Backwards Compatibility

### Old clients

Old clients would not be affected by the new Accept path.

If servers begin returning Status in version `api.k8s.io/v1`, old clients would likely error
as that group has never been used.  We would continue to return the group version of the calling
API group on server responses unless the `sv` mime-type parameter is set.


### Old servers

Because old Kubernetes servers are not selective about the content type parameters they
accept, we may wish to patch the 1.3 and 1.4 server versions to explicitly bypass content
types they do not recognize the parameters to. As a special consideration, this would allow
new clients to more strictly handle Accept (so that the server returns errors if the content
type is not recognized).

As part of introducing the new API group `api.k8s.io`, some opaque calls where we assume the
empty API group-version for the resource (GET parameters) could be defaulted to this group.


## Future items

* ???


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/job.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
