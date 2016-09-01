<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Protobuf serialization and internal storage

@smarterclayton

March 2016

## Proposal and Motivation

The Kubernetes API server is a "dumb server" which offers storage, versioning,
validation, update, and watch semantics on API resources. In a large cluster
the API server must efficiently retrieve, store, and deliver large numbers
of coarse-grained objects to many clients. In addition, Kubernetes traffic is
heavily biased towards intra-cluster traffic - as much as 90% of the requests
served by the APIs are for internal cluster components like nodes, controllers,
and proxies. The primary format for intercluster API communication is JSON
today for ease of client construction.

At the current time, the latency of reaction to change in the cluster is
dominated by the time required to load objects from persistent store (etcd),
convert them to an output version, serialize them JSON over the network, and
then perform the reverse operation in clients. The cost of
serialization/deserialization and the size of the bytes on the wire, as well
as the memory garbage created during those operations, dominate the CPU and
network usage of the API servers.

In order to reach clusters of 10k nodes, we need roughly an order of magnitude
efficiency improvement in a number of areas of the cluster, starting with the
masters but also including API clients like controllers, kubelets, and node
proxies.

We propose to introduce a Protobuf serialization for all common API objects
that can optionally be used by intra-cluster components. Experiments have
demonstrated a 10x reduction in CPU use during serialization and deserialization,
a 2x reduction in size in bytes on the wire, and a 6-9x reduction in the amount
of objects created on the heap during serialization. The Protobuf schema
for each object will be automatically generated from the external API Go structs
we use to serialize to JSON.

Benchmarking showed that the time spent on the server in a typical GET
resembles:

          etcd -> decode -> defaulting -> convert to internal ->
    JSON          50us      5us           15us
    Proto         5us
    JSON          150allocs               80allocs
    Proto         100allocs

          process -> convert to external -> encode -> client
    JSON             15us                   40us
    Proto                                   5us
    JSON             80allocs               100allocs
    Proto                                   4allocs

 Protobuf has a huge benefit on encoding because it does not need to allocate
 temporary objects, just one large buffer. Changing to protobuf moves our
 hotspot back to conversion, not serialization.


## Design Points

* Generate Protobuf schema from Go structs (like we do for JSON) to avoid
  manual schema update and drift
* Generate Protobuf schema that is field equivalent to the JSON fields (no
  special types or enumerations), reducing drift for clients across formats.
* Follow our existing API versioning rules (backwards compatible in major
  API versions, breaking changes across major versions) by creating one
  Protobuf schema per API type.
* Continue to use the existing REST API patterns but offer an alternative
  serialization, which means existing client and server tooling can remain
  the same while benefiting from faster decoding.
* Protobuf objects on disk or in etcd will need to be self identifying at
  rest, like JSON, in order for backwards compatibility in storage to work,
  so we must add an envelope with apiVersion and kind to wrap the nested
  object, and make the data format recognizable to clients.
* Use the [gogo-protobuf](https://github.com/gogo/protobuf) Golang library to generate marshal/unmarshal
  operations, allowing us to bypass the expensive reflection used by the
  golang JSOn operation


## Alternatives

* We considered JSON compression to reduce size on wire, but that does not
  reduce the amount of memory garbage created during serialization and
  deserialization.
* More efficient formats like Msgpack were considered, but they only offer
  2x speed up vs the 10x observed for Protobuf
* gRPC was considered, but is a larger change that requires more core
  refactoring. This approach does not eliminate the possibility of switching
  to gRPC in the future.
* We considered attempting to improve JSON serialization, but the cost of
  implementing a more efficient serializer library than ugorji is
  significantly higher than creating a protobuf schema from our Go structs.


## Schema

The Protobuf schema for each API group and version will be generated from
the objects in that API group and version. The schema will be named using
the package identifier of the Go package, i.e.

    k8s.io/kubernetes/pkg/api/v1

Each top level object will be generated as a Protobuf message, i.e.:

    type Pod struct { ... }

    message Pod {}

Since the Go structs are designed to be serialized to JSON (with only the
int, string, bool, map, and array primitive types), we will use the
canonical JSON serialization as the protobuf field type wherever possible,
i.e.:

    JSON      Protobuf
    string -> string
    int    -> varint
    bool   -> bool
    array  -> repeating message|primitive

We disallow the use of the Go `int` type in external fields because it is
ambiguous depending on compiler platform, and instead always use `int32` or
`int64`.

We will use maps (a protobuf 3 extension that can serialize to protobuf 2)
to represent JSON maps:

    JSON      Protobuf            Wire (proto2)
    map    -> map<string, ...> -> repeated Message { key string; value bytes }

We will not convert known string constants to enumerations, since that
would require extra logic we do not already have in JSOn.

To begin with, we will use Protobuf 3 to generate a Protobuf 2 schema, and
in the future investigate a Protobuf 3 serialization. We will introduce
abstractions that let us have more than a single protobuf serialization if
necessary. Protobuf 3 would require us to support message types for
pointer primitive (nullable) fields, which is more complex than Protobuf 2's
support for pointers.

### Example of generated proto IDL

Without gogo extensions:

```
syntax = 'proto2';

package k8s.io.kubernetes.pkg.api.v1;

import "k8s.io/kubernetes/pkg/api/resource/generated.proto";
import "k8s.io/kubernetes/pkg/api/unversioned/generated.proto";
import "k8s.io/kubernetes/pkg/runtime/generated.proto";
import "k8s.io/kubernetes/pkg/util/intstr/generated.proto";

// Package-wide variables from generator "generated".
option go_package = "v1";

// Represents a Persistent Disk resource in AWS.
//
// An AWS EBS disk must exist before mounting to a container. The disk
// must also be in the same AWS zone as the kubelet. An AWS EBS disk
// can only be mounted as read/write once. AWS EBS volumes support
// ownership management and SELinux relabeling.
message AWSElasticBlockStoreVolumeSource {
  // Unique ID of the persistent disk resource in AWS (Amazon EBS volume).
  // More info: http://releases.k8s.io/release-1.4/docs/user-guide/volumes.md#awselasticblockstore
  optional string volumeID = 1;

  // Filesystem type of the volume that you want to mount.
  // Tip: Ensure that the filesystem type is supported by the host operating system.
  // Examples: "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
  // More info: http://releases.k8s.io/release-1.4/docs/user-guide/volumes.md#awselasticblockstore
  // TODO: how do we prevent errors in the filesystem from compromising the machine
  optional string fsType = 2;

  // The partition in the volume that you want to mount.
  // If omitted, the default is to mount by volume name.
  // Examples: For volume /dev/sda1, you specify the partition as "1".
  // Similarly, the volume partition for /dev/sda is "0" (or you can leave the property empty).
  optional int32 partition = 3;

  // Specify "true" to force and set the ReadOnly property in VolumeMounts to "true".
  // If omitted, the default is "false".
  // More info: http://releases.k8s.io/release-1.4/docs/user-guide/volumes.md#awselasticblockstore
  optional bool readOnly = 4;
}

// Affinity is a group of affinity scheduling rules, currently
// only node affinity, but in the future also inter-pod affinity.
message Affinity {
  // Describes node affinity scheduling rules for the pod.
  optional NodeAffinity nodeAffinity = 1;
}
```

With extensions:

```
syntax = 'proto2';

package k8s.io.kubernetes.pkg.api.v1;

import "github.com/gogo/protobuf/gogoproto/gogo.proto";
import "k8s.io/kubernetes/pkg/api/resource/generated.proto";
import "k8s.io/kubernetes/pkg/api/unversioned/generated.proto";
import "k8s.io/kubernetes/pkg/runtime/generated.proto";
import "k8s.io/kubernetes/pkg/util/intstr/generated.proto";

// Package-wide variables from generator "generated".
option (gogoproto.marshaler_all) = true;
option (gogoproto.sizer_all) = true;
option (gogoproto.unmarshaler_all) = true;
option (gogoproto.goproto_unrecognized_all) = false;
option (gogoproto.goproto_enum_prefix_all) = false;
option (gogoproto.goproto_getters_all) = false;
option go_package = "v1";

// Represents a Persistent Disk resource in AWS.
//
// An AWS EBS disk must exist before mounting to a container. The disk
// must also be in the same AWS zone as the kubelet. An AWS EBS disk
// can only be mounted as read/write once. AWS EBS volumes support
// ownership management and SELinux relabeling.
message AWSElasticBlockStoreVolumeSource {
  // Unique ID of the persistent disk resource in AWS (Amazon EBS volume).
  // More info: http://releases.k8s.io/release-1.4/docs/user-guide/volumes.md#awselasticblockstore
  optional string volumeID = 1 [(gogoproto.customname) = "VolumeID", (gogoproto.nullable) = false];

  // Filesystem type of the volume that you want to mount.
  // Tip: Ensure that the filesystem type is supported by the host operating system.
  // Examples: "ext4", "xfs", "ntfs". Implicitly inferred to be "ext4" if unspecified.
  // More info: http://releases.k8s.io/release-1.4/docs/user-guide/volumes.md#awselasticblockstore
  // TODO: how do we prevent errors in the filesystem from compromising the machine
  optional string fsType = 2 [(gogoproto.customname) = "FSType", (gogoproto.nullable) = false];

  // The partition in the volume that you want to mount.
  // If omitted, the default is to mount by volume name.
  // Examples: For volume /dev/sda1, you specify the partition as "1".
  // Similarly, the volume partition for /dev/sda is "0" (or you can leave the property empty).
  optional int32 partition = 3 [(gogoproto.customname) = "Partition", (gogoproto.nullable) = false];

  // Specify "true" to force and set the ReadOnly property in VolumeMounts to "true".
  // If omitted, the default is "false".
  // More info: http://releases.k8s.io/release-1.4/docs/user-guide/volumes.md#awselasticblockstore
  optional bool readOnly = 4 [(gogoproto.customname) = "ReadOnly", (gogoproto.nullable) = false];
}

// Affinity is a group of affinity scheduling rules, currently
// only node affinity, but in the future also inter-pod affinity.
message Affinity {
  // Describes node affinity scheduling rules for the pod.
  optional NodeAffinity nodeAffinity = 1 [(gogoproto.customname) = "NodeAffinity"];
}
```

## Wire format

In order to make Protobuf serialized objects recognizable in a binary form,
the encoded object must be prefixed by a magic number, and then wrap the
non-self-describing Protobuf object in a Protobuf object that contains
schema information.  The protobuf object is referred to as the `raw` object
and the encapsulation is referred to as `wrapper` object.

The simplest serialization is the raw Protobuf object with no identifying
information. In some use cases, we may wish to have the server identify the
raw object type on the wire using a protocol dependent format (gRPC uses
a type HTTP header). This works when all objects are of the same type, but
we occasionally have reasons to encode different object types in the same
context (watches, lists of objects on disk, and API calls that may return
errors).

To identify the type of a wrapped Protobuf object, we wrap it in a message
in package `k8s.io/kubernetes/pkg/runtime` with message name `Unknown`
having the following schema:

    message Unknown {
      optional TypeMeta typeMeta = 1;
      optional bytes value = 2;
      optional string contentEncoding = 3;
      optional string contentType = 4;
    }

    message TypeMeta {
      optional string apiVersion = 1;
      optional string kind = 2;
    }

The `value` field is an encoded protobuf object that matches the schema
defined in `typeMeta` and has optional `contentType` and `contentEncoding`
fields.  `contentType` and `contentEncoding` have the same meaning as in
HTTP, if unspecified `contentType` means "raw protobuf object", and
`contentEncoding` defaults to no encoding. If `contentEncoding` is
specified, the defined transformation should be applied to `value` before
attempting to decode the value.

The `contentType` field is required to support objects without a defined
protobuf schema, like the ThirdPartyResource or templates. Those objects
would have to be encoded as JSON or another structure compatible form
when used with Protobuf. Generic clients must deal with the possibility
that the returned value is not in the known type.

We add the `contentEncoding` field here to preserve room for future
optimizations like encryption-at-rest or compression of the nested content.
Clients should error when receiving an encoding they do not support.
Negotioting encoding is not defined here, but introducing new encodings
is similar to introducing a schema change or new API version.

A client should use the `kind` and `apiVersion` fields to identify the
correct protobuf IDL for that message and version, and then decode the
`bytes` field into that Protobuf message.

Any Unknown value written to stable storage will be given a 4 byte prefix
`0x6b, 0x38, 0x73, 0x00`, which correspond to `k8s` followed by a zero byte.
The content-type `application/vnd.kubernetes.protobuf` is defined as
representing the following schema:

    MESSAGE = '0x6b 0x38 0x73 0x00' UNKNOWN
    UNKNOWN = <protobuf serialization of k8s.io/kubernetes/pkg/runtime#Unknown>

A client should check for the first four bytes, then perform a protobuf
deserialization of the remaining bytes into the `runtime.Unknown` type.

## Streaming wire format

While the majority of Kubernetes APIs return single objects that can vary
in type (Pod vs Status, PodList vs Status), the watch APIs return a stream
of identical objects (Events). At the time of this writing, this is the only
current or anticipated streaming RESTful protocol (logging, port-forwarding,
and exec protocols use a binary protocol over Websockets or SPDY).

In JSON, this API is implemented as a stream of JSON objects that are
separated by their syntax (the closing `}` brace is followed by whitespace
and the opening `{` brace starts the next object). There is no formal
specification covering this pattern, nor a unique content-type. Each object
is expected to be of type `watch.Event`, and is currently not self describing.

For expediency and consistency, we define a format for Protobuf watch Events
that is similar. Since protobuf messages are not self describing, we must
identify the boundaries between Events (a `frame`). We do that by prefixing
each frame of N bytes with a 4-byte, big-endian, unsigned integer with the
value N.

    frame  = length body
    length = 32-bit unsigned integer in big-endian order, denoting length of
             bytes of body
    body = <bytes>

    # frame containing a single byte 0a
    frame = 01 00 00 00 0a

    # equivalent JSON
    frame = {"type": "added", ...}

The body of each frame is a serialized Protobuf message `Event` in package
`k8s.io/kubernetes/pkg/watch/versioned`. The content type used for this
format is `application/vnd.kubernetes.protobuf;type=watch`.

## Negotiation

To allow clients to request protobuf serialization optionally, the `Accept`
HTTP header is used by callers to indicate which serialization they wish
returned in the response, and the `Content-Type` header is used to tell the
server how to decode the bytes sent in the request (for DELETE/POST/PUT/PATCH
requests). The server will return 406 if the `Accept` header is not
recognized or 415 if the `Content-Type` is not recognized (as defined in
RFC2616).

To be backwards compatible, clients must consider that the server does not
support protobuf serialization. A number of options are possible:

### Preconfigured

Clients can have a configuration setting that instructs them which version
to use. This is the simplest option, but requires intervention when the
component upgrades to protobuf.

### Include serialization information in api-discovery

Servers can define the list of content types they accept and return in
their API discovery docs, and clients can use protobuf if they support it.
Allows dynamic configuration during upgrade if the client is already using
API-discovery.

### Optimistically attempt to send and receive requests using protobuf

Using multiple `Accept` values:

    Accept: application/vnd.kubernetes.protobuf, application/json

clients can indicate their preferences and handle the returned
`Content-Type` using whatever the server responds. On update operations,
clients can try protobuf and if they receive a 415 error, record that and
fall back to JSON. Allows the client to be backwards compatible with
any server, but comes at the cost of some implementation complexity.


## Generation process

Generation proceeds in five phases:

1. Generate a gogo-protobuf annotated IDL from the source Go struct.
2. Generate temporary Go structs from the IDL using gogo-protobuf.
3. Generate marshaller/unmarshallers based on the IDL using gogo-protobuf.
4. Take all tag numbers generated for the IDL and apply them as struct tags
   to the original Go types.
5. Generate a final IDL without gogo-protobuf annotations as the canonical IDL.

The output is a `generated.proto` file in each package containing a standard
proto2 IDL, and a `generated.pb.go` file in each package that contains the
generated marshal/unmarshallers.

The Go struct generated by gogo-protobuf from the first IDL must be identical
to the origin struct - a number of changes have been made to gogo-protobuf
to ensure exact 1-1 conversion. A small number of additions may be necessary
in the future if we introduce more exotic field types (Go type aliases, maps
with aliased Go types, and embedded fields were fixed). If they are identical,
the output marshallers/unmarshallers can then work on the origin struct.

Whenever a new field is added, generation will assign that field a unique tag
and the 4th phase will write that tag back to the origin Go struct as a `protobuf`
struct tag. This ensures subsequent generation passes are stable, even in the
face of internal refactors. The first time a field is added, the author will
need to check in both the new IDL AND the protobuf struct tag changes.

The second IDL is generated without gogo-protobuf annotations to allow clients
in other languages to generate easily.

Any errors in the generation process are considered fatal and must be resolved
early (being unable to identify a field type for conversion, duplicate fields,
duplicate tags, protoc errors, etc). The conversion fuzzer is used to ensure
that a Go struct can be round-tripped to protobuf and back, as we do for JSON
and conversion testing.


## Changes to development process

All existing API change rules would still apply. New fields added would be
automatically assigned a tag by the generation process. New API versions will
have a new proto IDL, and field name and changes across API versions would be
handled using our existing API change rules. Tags cannot change within an
API version.

Generation would be done by developers and then checked into source control,
like conversions and ugorji JSON codecs.

Because protoc is not packaged well across all platforms, we will add it to
the `kube-cross` Docker image and developers can use that to generate
updated protobufs. Protobuf 3 beta is required.

The generated protobuf will be checked with a verify script before merging.


## Implications

* The generated marshal code is large and will increase build times and binary
  size. We may be able to remove ugorji after protobuf is added, since the
  bulk of our decoding would switch to protobuf.
* The protobuf schema is naive, which means it may not be as a minimal as
  possible.
* Debugging of protobuf related errors is harder due to the binary nature of
  the format.
* Migrating API object storage from JSON to protobuf will require that all
  API servers are upgraded before beginning to write protobuf to disk, since
  old servers won't recognize protobuf.
* Transport of protobuf between etcd and the api server will be less efficient
  in etcd2 than etcd3 (since etcd2 must encode binary values returned as JSON).
  Should still be smaller than current JSON request.
* Third-party API objects must be stored as JSON inside of a protobuf wrapper
  in etcd, and the API endpoints will not benefit from clients that speak
  protobuf. Clients will have to deal with some API objects not supporting
  protobuf.


## Open Questions

* Is supporting stored protobuf files on disk in the kubectl client worth it?




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/protobuf.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
