# Container Storage Interface (CSI)

Authors:

* Jie Yu <<jie@mesosphere.io>> (@jieyu)
* Saad Ali <<saadali@google.com>> (@saad-ali)
* James DeFelice <<james@mesosphere.io>> (@jdef)
* <container-storage-interface-working-group@googlegroups.com>

## Notational Conventions

The keywords "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" are to be interpreted as described in [RFC 2119](http://tools.ietf.org/html/rfc2119) (Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels", BCP 14, RFC 2119, March 1997).

The key words "unspecified", "undefined", and "implementation-defined" are to be interpreted as described in the [rationale for the C99 standard](http://www.open-std.org/jtc1/sc22/wg14/www/C99RationaleV5.10.pdf#page=18).

An implementation is not compliant if it fails to satisfy one or more of the MUST, REQUIRED, or SHALL requirements for the protocols it implements.
An implementation is compliant if it satisfies all the MUST, REQUIRED, and SHALL requirements for the protocols it implements.

## Terminology

| Term              | Definition                                       |
|-------------------|--------------------------------------------------|
| Volume            | A unit of storage that will be made available inside of a CO-managed container, via the CSI.                          |
| Block Volume      | A volume that will appear as a block device inside the container.                                                     |
| Mounted Volume    | A volume that will be mounted using the specified file system and appear as a directory inside the container.         |
| CO                | Container Orchestration system, communicates with Plugins using CSI service RPCs.                                     |
| SP                | Storage Provider, the vendor of a CSI plugin implementation.                                                          |
| RPC               | [Remote Procedure Call](https://en.wikipedia.org/wiki/Remote_procedure_call).                                         |
| Node              | A host where the user workload will be running, uniquely identifiable from the perspective of a Plugin by a node ID. |
| Plugin            | Aka “plugin implementation”, a gRPC endpoint that implements the CSI Services.                                        |
| Plugin Supervisor | Process that governs the lifecycle of a Plugin, MAY be the CO.                                                        |
| Workload          | The atomic unit of "work" scheduled by a CO. This may be a container or a collection of containers.                   |

## Objective

To define an industry standard “Container Storage Interface” (CSI) that will enable storage vendors (SP) to develop a plugin once and have it work across a number of container orchestration (CO) systems.

### Goals in MVP

The Container Storage Interface (CSI) will

* Enable SP authors to write one CSI compliant Plugin that “just works” across all COs that implement CSI.
* Define API (RPCs) that enable:
  * Dynamic provisioning and deprovisioning of a volume.
  * Attaching or detaching a volume from a node.
  * Mounting/unmounting a volume from a node.
  * Consumption of both block and mountable volumes.
  * Local storage providers (e.g., device mapper, lvm).
* Define plugin protocol RECOMMENDATIONS.
  * Describe a process by which a Supervisor configures a Plugin.
  * Container deployment considerations (`CAP_SYS_ADMIN`, mount namespace, etc.).

### Non-Goals in MVP

The Container Storage Interface (CSI) explicitly will not define, provide, or dictate in v0.1:

* Specific mechanisms by which a Plugin Supervisor manages the lifecycle of a Plugin, including:
  * How to maintain state (e.g. what is attached, mounted, etc.).
  * How to deploy, install, upgrade, uninstall, monitor, or respawn (in case of unexpected termination) Plugins.
* A first class message structure/field to represent "grades of storage" (aka "storage class").
* Protocol-level authentication and authorization.
* Packaging of a Plugin.
* POSIX compliance: CSI provides no guarantee that volumes provided are POSIX compliant filesystems.
  Compliance is determined by the Plugin implementation (and any backend storage system(s) upon which it depends).
  CSI SHALL NOT obstruct a Plugin Supervisor or CO from interacting with Plugin-managed volumes in a POSIX-compliant manner.

## Solution Overview

This specification defines an interface along with the minimum operational and packaging recommendations for a storage provider (SP) to implement a CSI compatible plugin.
The interface declares the RPCs that a plugin must expose: this is the **primary focus** of the CSI specification.
Any operational and packaging recommendations offer additional guidance to promote cross-CO compatibility.

### Architecture

The primary focus of this specification is on the **protocol** between a CO and a Plugin.
It SHOULD be possible to ship cross-CO compatible Plugins for a variety of deployment architectures.
A CO should be equipped to handle both centralized and headless plugins, as well as split-component and unified plugins.
Several of these possibilities are illustrated in the following figures.

```
                             CO "Master" Host
+-------------------------------------------+
|                                           |
|  +------------+           +------------+  |
|  |     CO     |   gRPC    | Controller |  |
|  |            +----------->   Plugin   |  |
|  +------------+           +------------+  |
|                                           |
+-------------------------------------------+

                            CO "Node" Host(s)
+-------------------------------------------+
|                                           |
|  +------------+           +------------+  |
|  |     CO     |   gRPC    |    Node    |  |
|  |            +----------->   Plugin   |  |
|  +------------+           +------------+  |
|                                           |
+-------------------------------------------+

Figure 1: The Plugin runs on all nodes in the cluster: a centralized
Controller Plugin is available on the CO master host and the Node
Plugin is available on all of the CO Nodes.
```

```
                            CO "Node" Host(s)
+-------------------------------------------+
|                                           |
|  +------------+           +------------+  |
|  |     CO     |   gRPC    | Controller |  |
|  |            +--+-------->   Plugin   |  |
|  +------------+  |        +------------+  |
|                  |                        |
|                  |                        |
|                  |        +------------+  |
|                  |        |    Node    |  |
|                  +-------->   Plugin   |  |
|                           +------------+  |
|                                           |
+-------------------------------------------+

Figure 2: Headless Plugin deployment, only the CO Node hosts run
Plugins. Separate, split-component Plugins supply the Controller
Service and the Node Service respectively.
```

```
                            CO "Node" Host(s)
+-------------------------------------------+
|                                           |
|  +------------+           +------------+  |
|  |     CO     |   gRPC    | Controller |  |
|  |            +----------->    Node    |  |
|  +------------+           |   Plugin   |  |
|                           +------------+  |
|                                           |
+-------------------------------------------+

Figure 3: Headless Plugin deployment, only the CO Node hosts run
Plugins. A unified Plugin component supplies both the Controller
Service and Node Service.
```

### Volume Lifecycle

```
   CreateVolume +------------+ DeleteVolume
 +------------->|  CREATED   +--------------+
 |              +---+----+---+              |
 |       Controller |    | Controller       v
+++         Publish |    | Unpublish       +++
|X|          Volume |    | Volume          | |
+-+             +---v----+---+             +-+
                | NODE_READY |
                +---+----^---+
               Node |    | Node
            Publish |    | Unpublish
             Volume |    | Volume
                +---v----+---+
                | PUBLISHED  |
                +------------+

Figure 4: The lifecycle of a dynamically provisioned volume, from
creation to destruction.
```

```
    Controller                  Controller
       Publish                  Unpublish
        Volume  +------------+  Volume
 +------------->+ NODE_READY +--------------+
 |              +---+----^---+              |
 |             Node |    | Node             v
+++         Publish |    | Unpublish       +++
|X| <-+      Volume |    | Volume          | |
+++   |         +---v----+---+             +-+
 |    |         | PUBLISHED  |
 |    |         +------------+
 +----+
   Validate
   Volume
   Capabilities

Figure 5: The lifecycle of a pre-provisioned volume that requires
controller to publish to a node (`ControllerPublishVolume`) prior to
publishing on the node (`NodePublishVolume`).
```

```
       +-+  +-+
       |X|  | |
       +++  +^+
        |    |
   Node |    | Node
Publish |    | Unpublish
 Volume |    | Volume
    +---v----+---+
    | PUBLISHED  |
    +------------+

Figure 6: Plugins may forego other lifecycle steps by contraindicating
them via the capabilities API. Interactions with the volumes of such
plugins is reduced to `NodePublishVolume` and `NodeUnpublishVolume`
calls.
```

The above diagrams illustrate a general expectation with respect to how a CO MAY manage the lifecycle of a volume via the API presented in this specification.
Plugins should expose all RPCs for an interface: Controller plugins should implement all RPCs for the `Controller` service.
Unsupported RPCs should return an appropriate error code that indicates such (e.g. `CALL_NOT_IMPLEMENTED`).
The full list of plugin capabilities is documented in the `ControllerGetCapabilities` and `NodeGetCapabilities` RPCs.

## Container Storage Interface

This section describes the interface between COs and Plugins.

### RPC Interface

A CO interacts with an Plugin through RPCs.
Each SP MUST provide:

* **Node Plugin**: A gRPC endpoint serving CSI RPCs that MUST be run on the Node whereupon an SP-provisioned volume will be published.
* **Controller Plugin**: A gRPC endpoint serving CSI RPCs that MAY be run anywhere.
* In some circumstances a single gRPC endpoint MAY serve all CSI RPCs (see Figure 3 in [Architecture](#architecture)).

```protobuf
syntax = "proto3";
package csi;
```

There are three sets of RPCs:

* **Identity Service**: Both the Node Plugin and the Controller Plugin MUST implement this sets of RPCs.
* **Controller Service**: The Controller Plugin MUST implement this sets of RPCs.
* **Node Service**: The Node Plugin MUST implement this sets of RPCs.

```protobuf
service Identity {
  rpc GetSupportedVersions (GetSupportedVersionsRequest)
    returns (GetSupportedVersionsResponse) {}

  rpc GetPluginInfo(GetPluginInfoRequest)
    returns (GetPluginInfoResponse) {}
}

service Controller {
  rpc CreateVolume (CreateVolumeRequest)
    returns (CreateVolumeResponse) {}

  rpc DeleteVolume (DeleteVolumeRequest)
    returns (DeleteVolumeResponse) {}

  rpc ControllerPublishVolume (ControllerPublishVolumeRequest)
    returns (ControllerPublishVolumeResponse) {}

  rpc ControllerUnpublishVolume (ControllerUnpublishVolumeRequest)
    returns (ControllerUnpublishVolumeResponse) {}

  rpc ValidateVolumeCapabilities (ValidateVolumeCapabilitiesRequest)
    returns (ValidateVolumeCapabilitiesResponse) {}

  rpc ListVolumes (ListVolumesRequest)
    returns (ListVolumesResponse) {}

  rpc GetCapacity (GetCapacityRequest)
    returns (GetCapacityResponse) {}

  rpc ControllerProbe (ControllerProbeRequest)
    returns (ControllerProbeResponse) {}

  rpc ControllerGetCapabilities (ControllerGetCapabilitiesRequest)
    returns (ControllerGetCapabilitiesResponse) {}  
}

service Node {
  rpc NodePublishVolume (NodePublishVolumeRequest)
    returns (NodePublishVolumeResponse) {}

  rpc NodeUnpublishVolume (NodeUnpublishVolumeRequest)
    returns (NodeUnpublishVolumeResponse) {}

  rpc GetNodeID (GetNodeIDRequest)
    returns (GetNodeIDResponse) {}

  rpc NodeProbe (NodeProbeRequest)
    returns (NodeProbeResponse) {}

  rpc NodeGetCapabilities (NodeGetCapabilitiesRequest)
    returns (NodeGetCapabilitiesResponse) {}
}
```

In general the Cluster Orchestrator (CO) is responsible for ensuring that there is no more than one call “in-flight” per volume at a given time.
However, in some circumstances, the CO may lose state (for example when the CO crashes and restarts), and may issue multiple calls simultaneously for the same volume.
The plugin should handle this as gracefully as possible.
The error code `OPERATION_PENDING_FOR_VOLUME` may be returned by the plugin in this case (see general error code section for details).

#### Field Size Limits

CSI defines general size limits for fields of various types (see table below).
The general size limit for a particular field may be overridden by specifying a different size limit in said field's description.
Unless otherwise specified, fields SHALL NOT exceed the limits documented here.
These limits apply for messages generated by both COs and plugins.

| Size       | Field Type          |
|------------|---------------------|
| 128 bytes  | string              |
| 4 KiB      | map<string, string> |

#### `REQUIRED` Fields

* A field listed as `REQUIRED` MUST be specified, subject to any per-RPC caveats; caveats SHOULD be rare.
* A `repeated` or `map` field listed as `REQUIRED` MUST contain at least 1 element.

### Error Scheme

All CSI API calls defined in this spec MUST return a [standard gRPC status](https://github.com/grpc/grpc/blob/master/src/proto/grpc/status/status.proto).
Most gRPC libraries provide helper methods to set and read the status fields.

The status `code` MUST contain a [canonical error code](https://github.com/grpc/grpc-go/blob/master/codes/codes.go). COs must handle all valid error codes. Each RPC defines a set of gRPC error codes that MUST be returned by the plugin when specified conditions are encountered. In addition to those, if the conditions defined below are encountered, the plugin MUST return the associated gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Unsupported request version | 3 INVALID_ARGUMENT | Indicates that the version specified in the request is not supported by the Plugin. | Caller MUST NOT retry; caller SHOULD call `GetSupportedVersions` to discover which CSI versions the Plugin supports. |
| Missing required field | 3 INVALID_ARGUMENT | Indicates that a required field is missing from the request. More human-readable information MAY be provided in the `error_description` field. | Caller MUST fix the request by adding the missing required field before retrying. |
| Invalid or unsupported field in the request | 3 INVALID_ARGUMENT | Indicates that the one ore more fields in this field is either not allowed by the Plugin or has an invalid value. More human-readable information MAY be provided in the gRPC `status.message` field. | Caller MUST fix the the field before retrying. |

The status `message` MUST contain a human readable description of error, if the status `code` is not `OK`.
This string MAY be surfaced by CO to end users.

The status `details` MUST be empty. In the future, this spec may require `details` to return a machine-parsable protobuf message if the status `code` is not `OK` to enable CO's to implement smarter error handling and fault resolution.

### Identity Service RPC

Identity service RPCs allow a CO to negotiate an API protocol version that MAY be used for subsequent RPCs across all CSI services with respect to a particular CSI plugin.
The general flow of the success case is as follows (protos illustrated in YAML for brevity):

1. CO queries supported versions via Identity RPC. The CO is expected to gracefully handle, in the manner of its own choosing, the case wherein the returned `supported_versions` from the plugin are not supported by the CO.

```
   # CO --(GetSupportedVersions)--> Plugin
   request: {}
   response:
      supported_versions:
        - major: 0
          minor: 1
          patch: 0
```

2. CO queries metadata via Identity RPC, using a supported API protocol version (as per the reply from the prior step): the requested `version` MUST match an entry from the aforementioned `supported_versions` array.

```
   # CO --(GetPluginInfo)--> Plugin
   request:
     version:
       major: 0
       minor: 1
       patch: 0
   response:
      name: org.foo.whizbang.super-plugin
      vendor_version: blue-green
      manifest:
        baz: qaz
```

#### `GetSupportedVersions`

A Plugin SHALL reply with a list of supported CSI versions.
The initial version of the CSI specification is 0.1.0 (in *major.minor.patch* format).
A CO MAY execute plugin RPCs in the manner prescribed by any such supported CSI version.
The versions returned by this call are orthogonal to any vendor-specific version metadata (see `vendor_version` in `GetPluginInfoResponse`).

NOTE: Changes to this RPC should be approached very conservatively since the request/response protobufs here are critical for proper client-server version negotiation.
Future changes to this RPC MUST **guarantee** backwards compatibility.

```protobuf
message GetSupportedVersionsRequest {
}

message GetSupportedVersionsResponse {
  // All the CSI versions that the Plugin supports. This field is
  // REQUIRED.
  repeated Version supported_versions = 1;
}

// Specifies a version in Semantic Version 2.0 format.
// (http://semver.org/spec/v2.0.0.html)
message Version {
  uint32 major = 1;  // This field is REQUIRED.
  uint32 minor = 2;  // This field is REQUIRED.
  uint32 patch = 3;  // This field is REQUIRED.
}
```

##### GetSupportedVersions Errors

If the plugin is unable to complete the GetSupportedVersions call successfully, it MUST return a non-ok gRPC code in the gRPC status.

#### `GetPluginInfo`

```protobuf
message GetPluginInfoRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;
}

message GetPluginInfoResponse {
  // The name MUST follow reverse domain name notation format
  // (https://en.wikipedia.org/wiki/Reverse_domain_name_notation).
  // It SHOULD include the plugin's host company name and the plugin
  // name, to minimize the possibility of collisions. It MUST be 63
  // characters or less, beginning and ending with an alphanumeric
  // character ([a-z0-9A-Z]) with dashes (-), underscores (_),
  // dots (.), and alphanumerics between. This field is REQUIRED.
  string name = 1;

  // This field is REQUIRED. Value of this field is opaque to the CO.
  string vendor_version = 2;

  // This field is OPTIONAL. Values are opaque to the CO.
  map<string, string> manifest = 3;
}
```

##### GetPluginInfo Errors

If the plugin is unable to complete the GetPluginInfo call successfully, it MUST return a non-ok gRPC code in the gRPC status.

### Controller Service RPC

#### `CreateVolume`

A Controller Plugin MUST implement this RPC call if it has `CREATE_DELETE_VOLUME` controller capability.
This RPC will be called by the CO to provision a new volume on behalf of a user (to be consumed as either a block device or a mounted filesystem).
This operation MUST be idempotent.

```protobuf
message CreateVolumeRequest {
  // The API version assumed by the CO. This field is REQUIRED.
  Version version = 1;

  // The suggested name for the storage space. This field is REQUIRED.
  // It serves two purposes:
  // 1) Idempotency - This name is generated by the CO to achieve
  //    idempotency. If `CreateVolume` fails, the volume may or may not
  //    be provisioned. In this case, the CO may call `CreateVolume`
  //    again, with the same name, to ensure the volume exists. The
  //    Plugin should ensure that multiple `CreateVolume` calls for the
  //    same name do not result in more than one piece of storage
  //    provisioned corresponding to that name. If a Plugin is unable to
  //    enforce idempotency, the CO's error recovery logic could result
  //    in multiple (unused) volumes being provisioned.
  // 2) Suggested name - Some storage systems allow callers to specify
  //    an identifier by which to refer to the newly provisioned
  //    storage. If a storage system supports this, it can optionally
  //    use this name as the identifier for the new volume.
  string name = 2;

  // This field is OPTIONAL. This allows the CO to specify the capacity
  // requirement of the volume to be provisioned. If not specified, the
  // Plugin MAY choose an implementation-defined capacity range.
  CapacityRange capacity_range = 3;

  // The capabilities that the provisioned volume MUST have: the Plugin
  // MUST provision a volume that could satisfy ALL of the
  // capabilities specified in this list. The Plugin MUST assume that
  // the CO MAY use the  provisioned volume later with ANY of the
  // capabilities specified in this list. This also enables the CO to do
  // early validation: if ANY of the specified volume capabilities are
  // not supported by the Plugin, the call SHALL fail. This field is
  // REQUIRED.
  repeated VolumeCapability volume_capabilities = 4;

  // Plugin specific parameters passed in as opaque key-value pairs.
  // This field is OPTIONAL. The Plugin is responsible for parsing and
  // validating these parameters. COs will treat these as opaque.
  map<string, string> parameters = 5;

  // End user credentials used to authenticate/authorize volume creation
  // request.
  // This field contains credential data, for example username and
  // password. Each key must consist of alphanumeric characters, '-',
  // '_' or '.'. Each value MUST contain a valid string. An SP MAY
  // choose to accept binary (non-string) data by using a binary-to-text
  // encoding scheme, like base64. An SP SHALL advertise the
  // requirements for credentials in documentation. COs SHALL permit
  // users to pass through the required credentials. This information is
  // sensitive and MUST be treated as such (not logged, etc.) by the CO.
  // This field is OPTIONAL.
  map<string, string> user_credentials = 6;
}

message CreateVolumeResponse {
  // Contains all attributes of the newly created volume that are
  // relevant to the CO along with information required by the Plugin
  // to uniquely identify the volume. This field is REQUIRED.
  VolumeInfo volume_info = 1;
}

// Specify a capability of a volume.
message VolumeCapability {
  // Indicate that the volume will be accessed via the block device API.
  message BlockVolume {
    // Intentionally empty, for now.
  }

  // Indicate that the volume will be accessed via the filesystem API.
  message MountVolume {
    // The filesystem type. This field is OPTIONAL.
    // An empty string is equal to an unspecified field value.
    string fs_type = 1;

    // The mount options that can be used for the volume. This field is
    // OPTIONAL. `mount_flags` MAY contain sensitive information.
    // Therefore, the CO and the Plugin MUST NOT leak this information
    // to untrusted entities. The total size of this repeated field
    // SHALL NOT exceed 4 KiB.
    repeated string mount_flags = 2;    
  }

  // Specify how a volume can be accessed.
  message AccessMode {
    enum Mode {
      UNKNOWN = 0;

      // Can be published as read/write at one node at a time.
      SINGLE_NODE_WRITER = 1;

      // Can be published as readonly at one node at a time.
      SINGLE_NODE_READER_ONLY = 2;

      // Can be published as readonly at multiple nodes simultaneously.
      MULTI_NODE_READER_ONLY = 3;

      // Can be published at multiple nodes simultaneously. Only one of
      // the node can be used as read/write. The rest will be readonly.
      MULTI_NODE_SINGLE_WRITER = 4;

      // Can be published as read/write at multiple nodes
      // simultaneously.
      MULTI_NODE_MULTI_WRITER = 5;
    }

    // This field is REQUIRED.
    Mode mode = 1;
  }

  // Specifies what API the volume will be accessed using. One of the
  // following fields MUST be specified.
  oneof access_type {
    BlockVolume block = 1;
    MountVolume mount = 2;
  }

  // This is a REQUIRED field.
  AccessMode access_mode = 3;
}

// The capacity of the storage space in bytes. To specify an exact size,
// `required_bytes` and `limit_bytes` can be set to the same value. At
// least one of the these fields MUST be specified.
message CapacityRange {
  // Volume must be at least this big. This field is OPTIONAL.
  // A value of 0 is equal to an unspecified field value.
  uint64 required_bytes = 1;

  // Volume must not be bigger than this. This field is OPTIONAL.
  // A value of 0 is equal to an unspecified field value.
  uint64 limit_bytes = 2;
}

// The information about a provisioned volume.
message VolumeInfo {
  // The capacity of the volume in bytes. This field is OPTIONAL. If not
  // set (value of 0), it indicates that the capacity of the volume is
  // unknown (e.g., NFS share).
  uint64 capacity_bytes = 1;

  // Contains identity information for the created volume. This field is
  // REQUIRED. The identity information will be used by the CO in
  // subsequent calls to refer to the provisioned volume.
  string id = 2;

  // Attributes reflect static properties of a volume and MUST be passed
  // to volume validation and publishing calls.
  // Attributes SHALL be opaque to a CO. Attributes SHALL NOT be mutable
  // and SHALL be safe for the CO to cache. Attributes SHOULD NOT
  // contain sensitive information. Attributes MAY NOT uniquely identify
  // a volume. A volume uniquely identified by `id` SHALL always report
  // the same attributes. This field is OPTIONAL and when present MUST
  // be passed to volume validation and publishing calls.
  map<string,string> attributes = 3;
}
```

##### CreateVolume Errors

If the plugin is unable to complete the CreateVolume call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Volume already exists | 6 ALREADY_EXISTS | Indicates that a volume corresponding to the specified volume `name` already exists. Plugin MUST also return a valid `CreateVolumeResponse`. | Caller MUST assume the `CreateVolume` call succeeded. |
| Operation pending for volume | 9 FAILED_PRECONDITION | Indicates that there is a already an operation pending for the specified volume. In general the Cluster Orchestrator (CO) is responsible for ensuring that there is no more than one call "in-flight" per volume at a given time. However, in some circumstances, the CO MAY lose state (for example when the CO crashes and restarts), and MAY issue multiple calls simultaneously for the same volume. The Plugin, SHOULD handle this as gracefully as possible, and MAY return this error code to reject secondary calls. | Caller SHOULD ensure that there are no other calls pending for the specified volume, and then retry with exponential back off. |
| Unsupported `capacity_range` | 11 OUT_OF_RANGE | Indicates that the capacity range is not allowed by the Plugin. More human-readable information MAY be provided in the gRPC `status.message` field. | Caller MUST fix the capacity range before retrying. |
| Call not implemented | 12 UNIMPLEMENTED | CreateVolume call is not implemented by the plugin or disabled in the Plugin's current mode of operation. | Caller MUST NOT retry. Caller MAY call `ControllerGetCapabilities` or `NodeGetCapabilities` to discover Plugin capabilities. |


#### `DeleteVolume`

A Controller Plugin MUST implement this RPC call if it has `CREATE_DELETE_VOLUME` capability.
This RPC will be called by the CO to deprovision a volume.
If successful, the storage space associated with the volume MUST be released and all the data in the volume SHALL NOT be accessible anymore.

This operation MUST be idempotent.
This operation SHOULD be best effort in the sense that if the Plugin is certain that the volume as well as the artifacts associated with the volume do not exist anymore, it SHOULD return a success.

```protobuf
message DeleteVolumeRequest {
  // The API version assumed by the CO. This field is REQUIRED.
  Version version = 1;

  // The ID of the volume to be deprovisioned.
  // This field is REQUIRED.
  string volume_id = 2;

  // End user credentials used to authenticate/authorize volume deletion
  // request.
  // This field contains credential data, for example username and
  // password. Each key must consist of alphanumeric characters, '-',
  // '_' or '.'. Each value MUST contain a valid string. An SP MAY
  // choose to accept binary (non-string) data by using a binary-to-text
  // encoding scheme, like base64. An SP SHALL advertise the
  // requirements for credentials in documentation. COs SHALL permit
  // users to pass through the required credentials. This information is
  // sensitive and MUST be treated as such (not logged, etc.) by the CO.
  // This field is OPTIONAL.
  map<string, string> user_credentials = 3;
}

message DeleteVolumeResponse {}
```

##### DeleteVolume Errors

If the plugin is unable to complete the DeleteVolume call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Volume does not exists | 5 NOT_FOUND | Indicates that a volume corresponding to the specified `volume_id` does not exist. | Caller MUST assume the `DeleteVolume` call succeeded. |
| Volume in use | 9 FAILED_PRECONDITION | Indicates that the volume corresponding to the specified `volume_id` could not be deleted because it is in use by another resource. | Caller SHOULD ensure that there are no other resources using the volume, and then retry with exponential back off. |
| Operation pending for volume | 9 FAILED_PRECONDITION | Indicates that there is a already an operation pending for the specified volume. In general the Cluster Orchestrator (CO) is responsible for ensuring that there is no more than one call "in-flight" per volume at a given time. However, in some circumstances, the CO MAY lose state (for example when the CO crashes and restarts), and MAY issue multiple calls simultaneously for the same volume. The Plugin, SHOULD handle this as gracefully as possible, and MAY return this error code to reject secondary calls. | Caller SHOULD ensure that there are no other calls pending for the specified volume, and then retry with exponential back off. |
| Call not implemented | 12 UNIMPLEMENTED | DeleteVolume call is not implemented by the plugin or disabled in the Plugin's current mode of operation. | Caller MUST NOT retry. Caller MAY call `ControllerGetCapabilities` or `NodeGetCapabilities` to discover Plugin capabilities. |


#### `ControllerPublishVolume`

A Controller Plugin MUST implement this RPC call if it has `PUBLISH_UNPUBLISH_VOLUME` controller capability.
This RPC will be called by the CO when it wants to place a workload that uses the volume onto a node.
The Plugin SHOULD perform the work that is necessary for making the volume available on the given node.
The Plugin MUST NOT assume that this RPC will be executed on the node where the volume will be used.

This operation MUST be idempotent.
If the operation failed or the CO does not know if the operation has failed or not, it MAY choose to call `ControllerPublishVolume` again or choose to call `ControllerUnpublishVolume`.

The CO MAY call this RPC for publishing a volume to multiple nodes if the volume has `MULTI_NODE` capability (i.e., `MULTI_NODE_READER_ONLY`, `MULTI_NODE_SINGLE_WRITER` or `MULTI_NODE_MULTI_WRITER`).

```protobuf
message ControllerPublishVolumeRequest {
  // The API version assumed by the CO. This field is REQUIRED.
  Version version = 1;

  // The ID of the volume to be used on a node.
  // This field is REQUIRED.
  string volume_id = 2;

  // The ID of the node. This field is REQUIRED. The CO SHALL set this
  // field to match the node ID returned by `GetNodeID`.
  string node_id = 3;

  // The capability of the volume the CO expects the volume to have.
  // This is a REQUIRED field.
  VolumeCapability volume_capability = 4;

  // Whether to publish the volume in readonly mode. This field is
  // REQUIRED.
  bool readonly = 5;

  // End user credentials used to authenticate/authorize controller
  // publish request.
  // This field contains credential data, for example username and
  // password. Each key must consist of alphanumeric characters, '-',
  // '_' or '.'. Each value MUST contain a valid string. An SP MAY
  // choose to accept binary (non-string) data by using a binary-to-text
  // encoding scheme, like base64. An SP SHALL advertise the
  // requirements for credentials in documentation. COs SHALL permit
  // users to pass through the required credentials. This information is
  // sensitive and MUST be treated as such (not logged, etc.) by the CO.
  // This field is OPTIONAL.
  map<string, string> user_credentials = 6;

  // Attributes of the volume to be used on a node. This field is
  // OPTIONAL and MUST match the attributes of the VolumeInfo identified
  // by `volume_id`.
  map<string,string> volume_attributes = 7;
}

message ControllerPublishVolumeResponse {
  // The SP specific information that will be passed to the Plugin in
  // the subsequent `NodePublishVolume` call for the given volume.
  // This information is opaque to the CO. This field is OPTIONAL.
  map<string, string> publish_volume_info = 1;
}
```

##### ControllerPublishVolume Errors

If the plugin is unable to complete the ControllerPublishVolume call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Volume does not exists | 5 NOT_FOUND | Indicates that a volume corresponding to the specified `volume_id` does not exist. | Caller MUST verify that the `volume_id` is correct and that the volume is accessible and has not been deleted before retrying with exponential back off. |
| Node does not exists | 5 NOT_FOUND | Indicates that a node corresponding to the specified `node_id` does not exist. | Caller MUST verify that the `node_id` is correct and that the node is available and has not been terminated or deleted before retrying with exponential backoff. |
| Volume published to another node | 6 ALREADY_EXISTS | Indicates that a volume corresponding to the specified `volume_id` is already attached to another node and does not support multi-node attach. If this error code is returned, the Plugin SHOULD specify the `node_id` of the node the volume is already attached to as part of the gRPC `status.message`. | Caller SHOULD ensure the specified volume is not attached to any other node before retrying with exponential back off. |
| Max volumes attached | 8 RESOURCE_EXHAUSTED | Indicates that the maximum supported number of volumes that can be attached to the specified node are already attached. Therefore, this operation will fail until at least one of the existing attached volumes is detached from the node. | Caller MUST ensure that the number of volumes already attached to the node is less then the maximum supported number of volumes before retrying with exponential backoff. |
| Operation pending for volume | 9 FAILED_PRECONDITION | Indicates that there is a already an operation pending for the specified volume. In general the Cluster Orchestrator (CO) is responsible for ensuring that there is no more than one call "in-flight" per volume at a given time. However, in some circumstances, the CO MAY lose state (for example when the CO crashes and restarts), and MAY issue multiple calls simultaneously for the same volume. The Plugin, SHOULD handle this as gracefully as possible, and MAY return this error code to reject secondary calls. | Caller SHOULD ensure that there are no other calls pending for the specified volume, and then retry with exponential back off. |
| Call not implemented | 12 UNIMPLEMENTED | ControllerPublishVolume call is not implemented by the plugin or disabled in the Plugin's current mode of operation. | Caller MUST NOT retry. Caller MAY call `ControllerGetCapabilities` or `NodeGetCapabilities` to discover Plugin capabilities. |

#### `ControllerUnpublishVolume`

Controller Plugin MUST implement this RPC call if it has `PUBLISH_UNPUBLISH_VOLUME` controller capability.
This RPC is a reverse operation of `ControllerPublishVolume`.
It MUST be called after `NodeUnpublishVolume` on the volume is called and succeeds.
The Plugin SHOULD perform the work that is necessary for making the volume ready to be consumed by a different node.
The Plugin MUST NOT assume that this RPC will be executed on the node where the volume was previously used.

This RPC is typically called by the CO when the workload using the volume is being moved to a different node, or all the workload using the volume on a node has finished.

This operation MUST be idempotent.
If this operation failed, or the CO does not know if the operation failed or not, it can choose to call `ControllerUnpublishVolume` again.

```protobuf
message ControllerUnpublishVolumeRequest {
  // The API version assumed by the CO. This field is REQUIRED.
  Version version = 1;

  // The ID of the volume. This field is REQUIRED.
  string volume_id = 2;

  // The ID of the node. This field is OPTIONAL. The CO SHOULD set this
  // field to match the node ID returned by `GetNodeID` or leave it
  // unset. If the value is set, the SP MUST unpublish the volume from
  // the specified node. If the value is unset, the SP MUST unpublish
  // the volume from all nodes it is published to.
  string node_id = 3;

  // End user credentials used to authenticate/authorize controller
  // unpublish request.
  // This field contains credential data, for example username and
  // password. Each key must consist of alphanumeric characters, '-',
  // '_' or '.'. Each value MUST contain a valid string. An SP MAY
  // choose to accept binary (non-string) data by using a binary-to-text
  // encoding scheme, like base64. An SP SHALL advertise the
  // requirements for credentials in documentation. COs SHALL permit
  // users to pass through the required credentials. This information is
  // sensitive and MUST be treated as such (not logged, etc.) by the CO.
  // This field is OPTIONAL.
  map<string, string> user_credentials = 4;
}

message ControllerUnpublishVolumeResponse {}
```

##### ControllerUnpublishVolume Errors

If the plugin is unable to complete the ControllerUnpublishVolume call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Volume not attached to specified node | 0 OK | Indicates that the volume corresponding to `volume_id` is not attached to the node corresponding to `node_id`. | Caller MUST assume the `ControllerUnpublishVolume` call succeeded. |
| Volume does not exists | 5 NOT_FOUND | Indicates that a volume corresponding to the specified `volume_id` does not exist. | Caller MUST verify that the `volume_id` is correct and that the volume is accessible and has not been deleted before retrying with exponential back off. |
| Node does not exists | 5 NOT_FOUND | Indicates that a node corresponding to the specified `node_id` does not exist. | Caller MUST verify that the `node_id` is correct and that the node is available and has not been terminated or deleted before retrying with exponential backoff. |
| Operation pending for volume | 9 FAILED_PRECONDITION | Indicates that there is a already an operation pending for the specified volume. In general the Cluster Orchestrator (CO) is responsible for ensuring that there is no more than one call "in-flight" per volume at a given time. However, in some circumstances, the CO MAY lose state (for example when the CO crashes and restarts), and MAY issue multiple calls simultaneously for the same volume. The Plugin, SHOULD handle this as gracefully as possible, and MAY return this error code to reject secondary calls. | Caller SHOULD ensure that there are no other calls pending for the specified volume, and then retry with exponential back off. |
| Call not implemented | 12 UNIMPLEMENTED | ControllerUnpublishVolume call is not implemented by the plugin or disabled in the Plugin's current mode of operation. | Caller MUST NOT retry. Caller MAY call `ControllerGetCapabilities` or `NodeGetCapabilities` to discover Plugin capabilities. |


#### `ValidateVolumeCapabilities`

A Controller Plugin MUST implement this RPC call.
This RPC will be called by the CO to check if a pre-provisioned volume has all the capabilities that the CO wants.
This RPC call SHALL return `supported` only if all the volume capabilities specified in the request are supported.
This operation MUST be idempotent.

```protobuf
message ValidateVolumeCapabilitiesRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;

  // The ID of the volume to check. This field is REQUIRED.
  string volume_id = 2;

  // The capabilities that the CO wants to check for the volume. This
  // call SHALL return "supported" only if all the volume capabilities
  // specified below are supported. This field is REQUIRED.
  repeated VolumeCapability volume_capabilities = 3;

  // Attributes of the volume to check. This field is OPTIONAL and MUST
  // match the attributes of the VolumeInfo identified by `volume_id`.
  map<string,string> volume_attributes = 4;
}

message ValidateVolumeCapabilitiesResponse {
  // True if the Plugin supports the specified capabilities for the
  // given volume. This field is REQUIRED.
  bool supported = 1;

  // Message to the CO if `supported` above is false. This field is
  // OPTIONAL.
  // An empty string is equal to an unspecified field value.
  string message = 2;
}
```

##### ValidateVolumeCapabilities Errors

If the plugin is unable to complete the ValidateVolumeCapabilities call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Volume does not exists | 5 NOT_FOUND | Indicates that a volume corresponding to the specified `volume_id` does not exist. | Caller MUST verify that the `volume_id` is correct and that the volume is accessible and has not been deleted before retrying with exponential back off. |

#### `ListVolumes`

A Controller Plugin MUST implement this RPC call if it has `LIST_VOLUMES` capability.
The Plugin SHALL return the information about all the volumes that it knows about.

```protobuf
message ListVolumesRequest {
  // The API version assumed by the CO. This field is REQUIRED.
  Version version = 1;

  // If specified (non-zero value), the Plugin MUST NOT return more
  // entries than this number in the response. If the actual number of
  // entries is more than this number, the Plugin MUST set `next_token`
  // in the response which can be used to get the next page of entries
  // in the subsequent `ListVolumes` call. This field is OPTIONAL. If
  // not specified (zero value), it means there is no restriction on the
  // number of entries that can be returned.
  uint32 max_entries = 2;

  // A token to specify where to start paginating. Set this field to
  // `next_token` returned by a previous `ListVolumes` call to get the
  // next page of entries. This field is OPTIONAL.
  // An empty string is equal to an unspecified field value.
  string starting_token = 3;
}

message ListVolumesResponse {
  message Entry {
    VolumeInfo volume_info = 1;
  }

  repeated Entry entries = 1;

  // This token allows you to get the next page of entries for
  // `ListVolumes` request. If the number of entries is larger than
  // `max_entries`, use the `next_token` as a value for the
  // `starting_token` field in the next `ListVolumes` request. This
  // field is OPTIONAL.
  // An empty string is equal to an unspecified field value.
  string next_token = 2;
}
```

##### ListVolumes Errors

If the plugin is unable to complete the ListVolumes call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Invalid `starting_token` | 10 ABORTED | Indicates that `starting_token` is not valid. | Caller SHOULD start the `ListVolumes` operation again with an empty `starting_token`. |

#### `GetCapacity`

A Controller Plugin MUST implement this RPC call if it has `GET_CAPACITY` controller capability.
The RPC allows the CO to query the capacity of the storage pool from which the controller provisions volumes.

```protobuf
message GetCapacityRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;

  // If specified, the Plugin SHALL report the capacity of the storage
  // that can be used to provision volumes that satisfy ALL of the
  // specified `volume_capabilities`. These are the same
  // `volume_capabilities` the CO will use in `CreateVolumeRequest`.
  // This field is OPTIONAL.
  repeated VolumeCapability volume_capabilities = 2;

  // If specified, the Plugin SHALL report the capacity of the storage
  // that can be used to provision volumes with the given Plugin
  // specific `parameters`. These are the same `parameters` the CO will
  // use in `CreateVolumeRequest`. This field is OPTIONAL.
  map<string, string> parameters = 3;
}

message GetCapacityResponse {
  // The available capacity of the storage that can be used to
  // provision volumes. If `volume_capabilities` or `parameters` is
  // specified in the request, the Plugin SHALL take those into
  // consideration when calculating the available capacity of the
  // storage. This field is REQUIRED.
  uint64 available_capacity = 1;
}
```

##### GetCapacity Errors

If the plugin is unable to complete the GetCapacity call successfully, it MUST return a non-ok gRPC code in the gRPC status.

#### `ControllerProbe`

A Controller Plugin MUST implement this RPC call.
The Plugin SHOULD verify if it has the right configurations, devices, dependencies and drivers in order to run the controller service, and return a success if the validation succeeds.
The CO SHALL invoke this RPC prior to any other controller service RPC in order to allow the CO to determine the readiness of the controller service.
A CO MAY invoke this call multiple times with the understanding that a plugin's implementation MAY NOT be trivial and there MAY be overhead incurred by such repeated calls.

```protobuf
message ControllerProbeRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;
}

message ControllerProbeResponse {}
```

##### ControllerProbe Errors

If the plugin is unable to complete the ControllerProbe call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Bad plugin config | 9 FAILED_PRECONDITION | Indicates that the plugin is misconfigured. | Caller MUST assume the plugin is not healthy. |
| Missing required dependency | 9 FAILED_PRECONDITION | Indicates that the plugin is missing one or more required dependency. | Caller MUST assume the plugin is not healthy. |

#### `ControllerGetCapabilities`

A Controller Plugin MUST implement this RPC call. This RPC allows the CO to check the supported capabilities of controller service provided by the Plugin.

```protobuf
message ControllerGetCapabilitiesRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;
}

message ControllerGetCapabilitiesResponse {
  // All the capabilities that the controller service supports. This
  // field is OPTIONAL.
  repeated ControllerServiceCapability capabilities = 2;
}

// Specifies a capability of the controller service.
message ControllerServiceCapability {
  message RPC {
    enum Type {
      UNKNOWN = 0;
      CREATE_DELETE_VOLUME = 1;
      PUBLISH_UNPUBLISH_VOLUME = 2;
      LIST_VOLUMES = 3;
      GET_CAPACITY = 4;
    }

    Type type = 1;
  }

  oneof type {
    // RPC that the controller supports.
    RPC rpc = 1;
  }
}
```

##### ControllerGetCapabilities Errors

If the plugin is unable to complete the ControllerGetCapabilities call successfully, it MUST return a non-ok gRPC code in the gRPC status.

### Node Service RPC

#### `NodePublishVolume`

This RPC is called by the CO when a workload that wants to use the specified volume is placed (scheduled) on a node.
The Plugin SHALL assume that this RPC will be executed on the node where the volume will be used.
This RPC MAY be called by the CO multiple times on the same node for the same volume with possibly different `target_path` and/or auth credentials.
If the corresponding Controller Plugin has `PUBLISH_UNPUBLISH_VOLUME` controller capability, the CO MUST guarantee that this RPC is called after `ControllerPublishVolume` is called for the given volume on the given node and returns a success.

This operation MUST be idempotent.
If this RPC failed, or the CO does not know if it failed or not, it MAY choose to call `NodePublishVolume` again, or choose to call `NodeUnpublishVolume`.

```protobuf
message NodePublishVolumeRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;

  // The ID of the volume to publish. This field is REQUIRED.
  string volume_id = 2;

  // The CO SHALL set this field to the value returned by
  // `ControllerPublishVolume` if the corresponding Controller Plugin
  // has `PUBLISH_UNPUBLISH_VOLUME` controller capability, and SHALL be
  // left unset if the corresponding Controller Plugin does not have
  // this capability. This is an OPTIONAL field.
  map<string, string> publish_volume_info = 3;

  // The path to which the volume will be published. It MUST be an
  // absolute path in the root filesystem of the process serving this
  // request. The CO SHALL ensure uniqueness of target_path per volume.
  // This is a REQUIRED field.
  string target_path = 4;

  // The capability of the volume the CO expects the volume to have.
  // This is a REQUIRED field.
  VolumeCapability volume_capability = 5;

  // Whether to publish the volume in readonly mode. This field is
  // REQUIRED.
  bool readonly = 6;

  // End user credentials used to authenticate/authorize node
  // publish request.
  // This field contains credential data, for example username and
  // password. Each key must consist of alphanumeric characters, '-',
  // '_' or '.'. Each value MUST contain a valid string. An SP MAY
  // choose to accept binary (non-string) data by using a binary-to-text
  // encoding scheme, like base64. An SP SHALL advertise the
  // requirements for credentials in documentation. COs SHALL permit
  // users to pass through the required credentials. This information is
  // sensitive and MUST be treated as such (not logged, etc.) by the CO.
  // This field is OPTIONAL.
  map<string, string> user_credentials = 7;

  // Attributes of the volume to publish. This field is OPTIONAL and
  // MUST match the attributes of the VolumeInfo identified by
  // `volume_id`.
  map<string,string> volume_attributes = 8;
}

message NodePublishVolumeResponse {}
```

##### NodePublishVolume Errors

If the plugin is unable to complete the NodePublishVolume call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Volume does not exists | 5 NOT_FOUND | Indicates that a volume corresponding to the specified `volume_id` does not exist. | Caller MUST verify that the `volume_id` is correct and that the volume is accessible and has not been deleted before retrying with exponential back off. |
| Operation pending for volume | 9 FAILED_PRECONDITION | Indicates that there is a already an operation pending for the specified volume. In general the Cluster Orchestrator (CO) is responsible for ensuring that there is no more than one call "in-flight" per volume at a given time. However, in some circumstances, the CO MAY lose state (for example when the CO crashes and restarts), and MAY issue multiple calls simultaneously for the same volume. The Plugin, SHOULD handle this as gracefully as possible, and MAY return this error code to reject secondary calls. | Caller SHOULD ensure that there are no other calls pending for the specified volume, and then retry with exponential back off. |

#### `NodeUnpublishVolume`

A Node Plugin MUST implement this RPC call.
This RPC is a reverse operation of `NodePublishVolume`.
This RPC MUST undo the work by the corresponding `NodePublishVolume`.
This RPC SHALL be called by the CO at least once for each `target_path` that was successfully setup via `NodePublishVolume`.
If the corresponding Controller Plugin has `PUBLISH_UNPUBLISH_VOLUME` controller capability, the CO SHOULD issue all `NodeUnpublishVolume` (as specified above) before calling `ControllerUnpublishVolume` for the given node and the given volume.
The Plugin SHALL assume that this RPC will be executed on the node where the volume is being used.

This RPC is typically called by the CO when the workload using the volume is being moved to a different node, or all the workload using the volume on a node has finished.

This operation MUST be idempotent.
If this RPC failed, or the CO does not know if it failed or not, it can choose to call `NodeUnpublishVolume` again.

```protobuf
message NodeUnpublishVolumeRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;

  // The ID of the volume. This field is REQUIRED.
  string volume_id = 2;

  // The path at which the volume was published. It MUST be an absolute
  // path in the root filesystem of the process serving this request.
  // This is a REQUIRED field.
  string target_path = 3;

  // End user credentials used to authenticate/authorize node
  // unpublish request.
  // This field contains credential data, for example username and
  // password. Each key must consist of alphanumeric characters, '-',
  // '_' or '.'. Each value MUST contain a valid string. An SP MAY
  // choose to accept binary (non-string) data by using a binary-to-text
  // encoding scheme, like base64. An SP SHALL advertise the
  // requirements for credentials in documentation. COs SHALL permit
  // users to pass through the required credentials. This information is
  // sensitive and MUST be treated as such (not logged, etc.) by the CO.
  // This field is OPTIONAL.
  map<string, string> user_credentials = 4;
}

message NodeUnpublishVolumeResponse {}
```

##### NodeUnpublishVolume Errors

If the plugin is unable to complete the NodeUnpublishVolume call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Volume does not exists | 5 NOT_FOUND | Indicates that a volume corresponding to the specified `volume_id` does not exist. | Caller MUST verify that the `volume_id` is correct and that the volume is accessible and has not been deleted before retrying with exponential back off. |
| Operation pending for volume | 9 FAILED_PRECONDITION | Indicates that there is a already an operation pending for the specified volume. In general the Cluster Orchestrator (CO) is responsible for ensuring that there is no more than one call "in-flight" per volume at a given time. However, in some circumstances, the CO MAY lose state (for example when the CO crashes and restarts), and MAY issue multiple calls simultaneously for the same volume. The Plugin, SHOULD handle this as gracefully as possible, and MAY return this error code to reject secondary calls. | Caller SHOULD ensure that there are no other calls pending for the specified volume, and then retry with exponential back off. |

#### `GetNodeID`

A Node Plugin MUST implement this RPC call if the plugin has `PUBLISH_UNPUBLISH_VOLUME` controller capability.
The Plugin SHALL assume that this RPC will be executed on the node where the volume will be used.
The CO SHOULD call this RPC for the node at which it wants to place the workload.
The result of this call will be used by CO in `ControllerPublishVolume`.

```protobuf
message GetNodeIDRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;
}

message GetNodeIDResponse {
  // The ID of the node as understood by the SP which SHALL be used by
  // CO in subsequent `ControllerPublishVolume`.
  // This is a REQUIRED field.
  string node_id = 1;
}
```

##### GetNodeID Errors

If the plugin is unable to complete the GetNodeID call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Call not implemented | 12 UNIMPLEMENTED | GetNodeID call is not implemented by the plugin or disabled in the Plugin's current mode of operation. | Caller MUST NOT retry. Caller MAY call `ControllerGetCapabilities` or `NodeGetCapabilities` to discover Plugin capabilities. |

#### `NodeProbe`

A Node Plugin MUST implement this RPC call.
The Plugin SHALL assume that this RPC will be executed on the node where the volume will be used.
The CO SHOULD call this RPC for the node at which it wants to place the workload.
This RPC allows the CO to probe the readiness of the Plugin on the node where the volumes will be used.
The Plugin SHOULD verify if it has everything it needs (binaries, kernel module, drivers, etc.) to run on that node, and return a success if the validation succeeds.
The CO MAY use this RPC to probe which machines can support specific Plugins and schedule workloads accordingly.

```protobuf
message NodeProbeRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;
}

message NodeProbeResponse {}
```

##### NodeProbe Errors

If the plugin is unable to complete the NodeProbe call successfully, it MUST return a non-ok gRPC code in the gRPC status.
If the conditions defined below are encountered, the plugin MUST return the specified gRPC error code.
The CO MUST implement the specified error recovery behavior when it encounters the gRPC error code.

Condition | gRPC Code | Description | Recovery Behavior
| --- | --- | --- | --- |
| Bad plugin config | 9 FAILED_PRECONDITION | Indicates that the plugin is misconfigured. | Caller MUST assume the plugin is not healthy. |
| Missing required dependency | 9 FAILED_PRECONDITION | Indicates that the plugin is missing one or more required dependency. | Caller MUST assume the plugin is not healthy. |

#### `NodeGetCapabilities`

A Node Plugin MUST implement this RPC call.
This RPC allows the CO to check the supported capabilities of node service provided by the Plugin.

```protobuf
message NodeGetCapabilitiesRequest {
  // The API version assumed by the CO. This is a REQUIRED field.
  Version version = 1;
}

message NodeGetCapabilitiesResponse {
  // All the capabilities that the node service supports. This field
  // is OPTIONAL.
  repeated NodeServiceCapability capabilities = 1;
}

// Specifies a capability of the node service.
message NodeServiceCapability {
  message RPC {
    enum Type {
      UNKNOWN = 0;
    }

    Type type = 1;
  }

  oneof type {
    // RPC that the controller supports.
    RPC rpc = 1;
  }
}
```

##### NodeGetCapabilities Errors

If the plugin is unable to complete the NodeGetCapabilities call successfully, it MUST return a non-ok gRPC code in the gRPC status.

## Protocol

### Connectivity

* A CO SHALL communicate with a Plugin using gRPC to access the `Identity`, and (optionally) the `Controller` and `Node` services.
  * proto3 SHOULD be used with gRPC, as per the [official recommendations](http://www.grpc.io/docs/guides/#protocol-buffer-versions).
  * All Plugins SHALL implement the REQUIRED Identity service RPCs.
    Support for OPTIONAL RPCs is reported by the `ControllerGetCapabilities` and `NodeGetCapabilities` RPC calls.
* The CO SHALL provide the listen-address for the Plugin by way of the `CSI_ENDPOINT` environment variable.
  Plugin components SHALL create, bind, and listen for RPCs on the specified listen address.
  * Only UNIX Domain Sockets may be used as endpoints.
    This will likely change in a future version of this specification to support non-UNIX platforms.
* All supported RPC services MUST be available at the listen address of the Plugin.

### Security

* The CO operator and Plugin Supervisor SHOULD take steps to ensure that any and all communication between the CO and Plugin Service are secured according to best practices.
* Communication between a CO and a Plugin SHALL be transported over UNIX Domain Sockets.
  * gRPC is compatible with UNIX Domain Sockets; it is the responsibility of the CO operator and Plugin Supervisor to properly secure access to the Domain Socket using OS filesystem ACLs and/or other OS-specific security context tooling.
  * SP’s supplying stand-alone Plugin controller appliances, or other remote components that are incompatible with UNIX Domain Sockets must provide a software component that proxies communication between a UNIX Domain Socket and the remote component(s).
    Proxy components transporting communication over IP networks SHALL be responsible for securing communications over such networks.
* Both the CO and Plugin SHOULD avoid accidental leakage of sensitive information (such as redacting such information from log files).

### Debugging

* Debugging and tracing are supported by external, CSI-independent additions and extensions to gRPC APIs, such as [OpenTracing](https://github.com/grpc-ecosystem/grpc-opentracing).

## Configuration and Operation

### General Configuration

* The `CSI_ENDPOINT` environment variable SHALL be supplied to the Plugin by the Plugin Supervisor.
* An operator SHALL configure the CO to connect to the Plugin via the listen address identified by `CSI_ENDPOINT` variable.
* With exception to sensitive data, Plugin configuration SHOULD be specified by environment variables, whenever possible, instead of by command line flags or bind-mounted/injected files.


#### Plugin Bootstrap Example

* Supervisor -> Plugin: `CSI_ENDPOINT=unix:///path/to/unix/domain/socket.sock`.
* Operator -> CO: use plugin at endpoint `unix:///path/to/unix/domain/socket.sock`.
* CO: monitor `/path/to/unix/domain/socket.sock`.
* Plugin: read `CSI_ENDPOINT`, create UNIX socket at specified path, bind and listen.
* CO: observe that socket now exists, establish connection.
* CO: invoke `GetSupportedVersions`.

#### Filesystem

* Plugins SHALL NOT specify requirements that include or otherwise reference directories and/or files on the root filesystem of the CO.
* Plugins SHALL NOT create additional files or directories adjacent to the UNIX socket specified by `CSI_ENDPOINT`; violations of this requirement constitute "abuse".
  * The Plugin Supervisor is the ultimate authority of the directory in which the UNIX socket endpoint is created and MAY enforce policies to prevent and/or mitigate abuse of the directory by Plugins.

### Supervised Lifecycle Management

* For Plugins packaged in software form:
  * Plugin Packages SHOULD use a well-documented container image format (e.g., Docker, OCI).
  * The chosen package image format MAY expose configurable Plugin properties as environment variables, unless otherwise indicated in the section below.
    Variables so exposed SHOULD be assigned default values in the image manifest.
  * A Plugin Supervisor MAY programmatically evaluate or otherwise scan a Plugin Package’s image manifest in order to discover configurable environment variables.
  * A Plugin SHALL NOT assume that an operator or Plugin Supervisor will scan an image manifest for environment variables.

#### Environment Variables

* Variables defined by this specification SHALL be identifiable by their `CSI_` name prefix.
* Configuration properties not defined by the CSI specification SHALL NOT use the same `CSI_` name prefix; this prefix is reserved for common configuration properties defined by the CSI specification.
* The Plugin Supervisor SHOULD supply all recommended CSI environment variables to a Plugin.
* The Plugin Supervisor SHALL supply all required CSI environment variables to a Plugin.

##### `CSI_ENDPOINT`

Network endpoint at which a Plugin SHALL host CSI RPC services. The general format is:

    {scheme}://{authority}{endpoint}

The following address types SHALL be supported by Plugins:

    unix:///path/to/unix/socket.sock

Note: All UNIX endpoints SHALL end with `.sock`. See [gRPC Name Resolution](https://github.com/grpc/grpc/blob/master/doc/naming.md).  

This variable is REQUIRED.

#### Operational Recommendations

The Plugin Supervisor expects that a Plugin SHALL act as a long-running service vs. an on-demand, CLI-driven process.

Supervised plugins MAY be isolated and/or resource-bounded.

##### Logging

* Plugins SHOULD generate log messages to ONLY standard output and/or standard error.
  * In this case the Plugin Supervisor SHALL assume responsibility for all log lifecycle management.
* Plugin implementations that deviate from the above recommendation SHALL clearly and unambiguously document the following:
  * Logging configuration flags and/or variables, including working sample configurations.
  * Default log destination(s) (where do the logs go if no configuration is specified?)
  * Log lifecycle management ownership and related guidance (size limits, rate limits, rolling, archiving, expunging, etc.) applicable to the logging mechanism embedded within the Plugin.
* Plugins SHOULD NOT write potentially sensitive data to logs (e.g. `Credentials`).

##### Available Services

* Plugin Packages MAY support all or a subset of CSI services; service combinations MAY be configurable at runtime by the Plugin Supervisor.
* Misconfigured plugin software SHOULD fail-fast with an OS-appropriate error code.

##### Linux Capabilities

* Plugin Supervisor SHALL guarantee that plugins will have `CAP_SYS_ADMIN` capability on Linux when running on Nodes.
* Plugins SHOULD clearly document any additionally required capabilities and/or security context.

##### Namespaces

* A Plugin SHOULD NOT assume that it is in the same [Linux namespaces](https://en.wikipedia.org/wiki/Linux_namespaces) as the Plugin Supervisor.
  The CO MUST clearly document the [mount propagation](https://www.kernel.org/doc/Documentation/filesystems/sharedsubtree.txt) requirements for Node Plugins and the Plugin Supervisor SHALL satisfy the CO’s requirements.

##### Cgroup Isolation

* A Plugin MAY be constrained by cgroups.
* An operator or Plugin Supervisor MAY configure the devices cgroup subsystem to ensure that a Plugin may access requisite devices.
* A Plugin Supervisor MAY define resource limits for a Plugin.

##### Resource Requirements

* SPs SHOULD unambiguously document all of a Plugin’s resource requirements.
