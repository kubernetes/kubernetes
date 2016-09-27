# Flex Volume API Proposal
## Abstract
The goal of this proposal is to extend and enhance the existing flex volume storage plugin model (#13840). It bridges the gap with existing in-tree plugins and defines a stable api.

## Background:
The existing flex volume plugin enables administrators/vendors to create out-of-tree volume plugins, i.e., plugins which do not reside in Kubernetes project. But, it has the following gaps:
* It lacks support for the latest enhancements to in-tree plugins like dynamic provisioning and centralized attach-detach controller
* It does not define a stable driver api for administrators/vendors to work with.
* The current flex volume plugin deployment model is not suitable for environments like GCE & CoreOS, where access to the root file system is restricted. And in other environments the overall user experience is poor, as it requires manual installation of the plugin on every node and restart of kubelets & controller manager. This issue is covered in detail in another PR #32550.

### Goals:
The goals of this proposal are 
* Define new stable driver API.
* Add support for dynamic provisioning and centralized attach-detach controller and
* The biggest advantage of flex volume is it’s simplicity. It is just an adapter around the internal api. We will keep it simple in the new design and try to follow the same adapter pattern.

### Non Goals:
* This proposal does not the address the poor installation model of existing model. It is covered in another PR #32550.

## Terminology:
**Plugin**: refers to flex volume plugin.

**Plugin Driver API**: API calls defined by flex volume plugin.

**Plugin Driver**: Out-of-tree driver/plugin which implements plugin driver api.

**In-tree plugin**: Plugins which are included in Kubernetes project/source tree.

## API:
Existing flex volume plugin driver api is in alpha phase and are subject to change between Kubernetes versions. This section proposes stable api for plugin driver writers to work with.

### Proposal:
Proposes fewer APIs with stable request & response API objects.

###### Pros:
* Has standard request & response API objects.
* Supports versioning.
* Uses gRPC.
* Fewer calls.
* Simple and clean interface.

###### Cons:
* New API objects.

#### APIs

```protobuf

// Capabilities of the driver
message DriverCapabilities {
	string supported_read_write_modes = 1;
	enum AttachmentPolicy {
	    NONE = 0;  // Driver does not support attach/detach.
	    LOCAL = 1; // Driver supports local volume attachments from kubelet.
	    REMOTE = 2; // Driver supports remote volume attachments to a node from controller-manager.
	}
	AttachmentPolicy supported_attachment_policy = 2;
	bool supports_dynamic_provisioning = 3;
	bool supports_custom_mount = 4;
	bool supports_selinux = 5;
	bool supports_ownership_management = 6;
	bool supports_metrics = 7;
}

message ProbeDriverRequest {}

message ProbeDriverResponse {
    DriverCapabilities capabilities = 1;
    repeated string options = 2;
}

service FlexVolumeService {
    // Probe the driver capabilities.
    rpc ProbeDriver(ProbeDriverRequest) returns (ProbeDriverResponse) {}
    // Create a volume.
    rpc Create(CreateRequest) returns (CreateResponse) {}
    // Delete a volume.
    rpc Delete(DeleteRequest) returns (DeleteResponse) {}
    // Attach the volume.
    rpc Attach(AttachRequest) returns (AttachResponse) {}
    // Detach the volume.
    rpc Detach(DetachRequest) returns (DetachResponse) {}
    // Mount the volume.
    rpc Mount(MountRequest) returns (MountResponse) {}
    // unmount the volume.
    rpc Unmount(UnmountRequest) returns (UnmountResponse) {}
    // Query the metrics.
    rpc GetMetrics(MetricsRequest) returns (MetricsResponse) {}
}

// Volume source.
message Spec {
    string name = 1; // Name of the volume.
    string fstype = 2; // Requested file system type.
    string read_write_mode = 3;
    map<string, string> secrets = 4; // Secrets required for the driver to talk to its own controller.
    map<string, string> options = 5; // Extra options passed to the driver.
}

message CreateRequest{
    Spec spec = 1; // Volume specification.
}

message CreateResponse{}

message DeleteRequest{
    string name = 1; // Name of the volume.
}

message DeleteResponse{}

message AttachRequest {
    string name = 1; // Name of the volume.
    string host = 2; // Name of the host to attach volume to.
    bool sync = 3; // Wait for attach to finish.
    map<string, string> secrets = 4; // Secrets required for the driver to talk to its own controller.
    map<string, string> options = 5; // Extra options passed to the driver.
}

message AttachResponse{
    string device_path = 1; // Path to the device where the volume is attached.
}

message DetachRequest {
    string name = 1; // Name of the volume.
    string host = 2; // Name of the host to detach volume from.
    bool sync = 3; // Wait for detach to finish.
    map<string, string> secrets = 4; // Extra options passed to the driver.
}

message DetachResponse {}

message MountRequest {
    Spec spec = 1; // Volume specifciation.
    string mount_path = 2; // Recommended path for the driver to mount the volume.
}

message MountResponse {
    string mount_path = 1; // Path where the driver actually mounted the volume.
}

message UnmountRequest {
    string mount_path = 1;
}

message UnmountResponse{}

message MetricsRequest {}

message MetricsResponse {
    Quantity capacity = 1;
    Quantity used = 2;
    Quantity avaialble = 3;
}

```

### Other options considered:
Expose the existing in-tree plugin API as driver API. For more details on existing in-tree plugin API please refer to https://github.com/kubernetes/kubernetes/blob/master/pkg/volume/volume.go

###### Pros:
* In-tree plugins are already using these API.
* Simple and clean interface.

###### Cons:
* No standard request & response API objects.
* No versioning support.
* Does not follow REST model.
* Many call outs. For full support, a plugin has to implement 17 APIs.

## Dynamic Provisioning support:
Dynamic provisioning support is optional. Driver can choose to implement it by implementing Create & Delete APIs. Driver should also report it supports dynamic provisioning by setting the "supports_dynamic_provisioning" capability.

## Attach detach support:
Attach detach support is optional. Driver can choose to implement it by implementing Attach & Detach APIs. Driver should also report the type of attachment policy it supports. If the driver supports attach from Kubelet it should report AttachmentPolicy "LOCAL" and if it supports attaching from Central controller(controller-manager), it should report AttachmentPolicy "GLOBAL".

PR by @MikaelCluseau (#26926) would be leveraged for this change.
