# Flex Volume API Proposal
## Abstract
The goal of this proposal is to extend and enhance the existing flex volume storage plugin model (#13840). It bridges the gap with existing in-tree plugins and defines a stable api.

## Background:
Existing flex volume plugin enables administrators/vendors to create out-of-tree volume plugins, i.e., plugins which do not reside in Kubernetes project. But, it has the following gaps.
* It lacks support for the latest enhancements to in-tree plugins like dynamic provisioning and centralized attach-detach controller
* It does not define a stable driver api for administrators/vendors to work.
* The current flex volume plugin deployment model is not suitable for environments like GCE & CoreOS, where access to the root file system is restricted. And in other environments the overall user experience is poor, as it requires manual installation of the plugin on every node and restart of kubelets & controller manager. This issue is covered in detail in another PR.

### Goals:
The goals of this proposal are 
* Define new stable driver API.
* Add support for dynamic provisioning and centralized attach-detach controller and
* The biggest advantage of flex volume is it’s simplicity. It is just an adapter around the internal api. We will keep it simple in the new design and try to follow the same adapter pattern.

### Non Goals:
* This proposal does not the address the poor installation model of existing model. It is covered in another PR.
* This proposal uses the existing ‘exec’ plugin model. A rest callout model will be covered in another PR.

## Terminology:
**Plugin**: refers to flex volume plugin.
**Plugin Driver api**: API calls defined by flex volume plugin.
**Plugin Driver**: Out-of-tree driver/plugin which implements plugin driver api.
**In-tree plugin**: Plugins which are included in Kubernetes project/source tree.

## API:
Existing flex volume plugin driver api are in alpha phase and are subject to change between kubernetes versions. This section proposes stable api for plugin driver writers to work with.
### Proposal 1:
Expose the existing in-tree plugin API as driver API. For more details on existing in-tree plugin API please refer to https://github.com/kubernetes/kubernetes/blob/master/pkg/volume/volume.go

###### Pros:
* In-tree plugins are already using these API.
* Simple and clean interface.

###### Cons:
* No standard request & response API objects.
* No versioning support.
* Does not follow REST model.
* Many call outs. For full support, a plugin has to implement 17 APIs.

This was discussed in the storage-sig face to face and in lieu the following is proposed.

### Proposal 2:
Proposes fewer APIs with stable request & response API objects.

###### Pros:
* Has standard request & response API objects.
* Supports versioning.
* Follows REST model.
* Fewer calls.
* Simple and clean interface.

###### Cons:
* New API objects.

#### APIs

#### Probe Plugin:
This call probes the plugin for it’s capabilities and supported options. Supported options are used to validate and reject pod spec. This is executed from both the Controller-manager and Kubelet.

###### New API Types:
```go
type FlexVolumeDriverCapabilities struct {
	ReadWrite api.PersistentVolumeAccessMode
	DynamicProvisioning bool
	Attachment bool        // Do we need separate capability for remote & local attachments support? Some drivers like ISCSI do not support attachment from remote node (controller manager).
	SELinux bool
	OwnershipManagement bool
	Metrics bool
}

type FlexVolumeDriverSpec struct {
    Name string     // name of the driver. Ex: ganesh-nfs
    Driver string   // Actual driver path. Ex: ganesha/nfs
}
type FlexVolumeDriver struct {
    unversioned.TypeMeta
    v1.ObjectMeta
    Spec FlexVolumeDriverSpec
    Capabilities FlexVolumeDriverCapabilities
    SupportedOptions []string // Driver options supported.
}
```

###### Request:
```
nil
```

###### Response:
On success:
```
FlexVolumeDriver
```
On failure:
```
non-zero exit code & errors.StatusError
```

###### Call out:
```
<driver executable> probe
```

#### Provision/Create a volume:
This call creates a volume. It is executed from Controller-manager.

###### New API Types:
```go
Type FlexVolumeStatus string

type FlexVolume struct {
    unversioned.TypeMeta
    v1.ObjectMeta
    Spec api.FlexVolumeSource
    Status FlexVolumeStatus
}
```

###### Request:
```
FlexVolume
```

###### Response:
On success:
```
api.FlexVolume
```
On failure:
```
non-zero exit code & errors.StatusError
```

###### Call out:
```
<driver executable> create <json encoded api.FlexVolume>
```

#### Delete a volume:
This call deletes a volume. It is executed from Controller-manager.

###### Request:
```
FlexVolume
```

###### Response:
On success:
```
0 exit code
```
On failure:
```
non-zero exit code & errors.StatusError
```

###### Call out:
```
<driver executable> delete <json encoded api.FlexVolume>
```

#### Attach a volume:
This call attaches a volume to a remote host or local host. It can be executed from Controller-manager/Kubelet depending on whether Controller-attach-detach is enabled or not.. This is only valid for drivers which support attach & detach.

The following additional options are passed to the plugin as part of options map in FlexVolumeSource.
```
“kubernetes.io/host”=”<host-name>”
“kubernetes.io/mountpath”=”<mount path>”
```

###### New API Types:
```go
type FlexVolumeAttachment struct {
    unversioned.TypeMeta
    v1.ObjectMeta
    Host string
    Device string
    MountPath string
}
```

###### Request:
```
FlexVolume
```

###### Response:
On success:
```
FlexVolumeAttachment
```
On failure:
```
non-zero exit code & errors.StatusError
```

###### Call out:
```
<driver executable> attach <json encoded FlexVolume>
```

###### Call out sync: (Wait for attach):
```
<driver executable> attach <json encoded FlexVolume> sync
```

#### Detach a volume:
This call detaches a volume to a remote host or local host. It can be executed from Controller-manager/Kubelet depending on whether Controller-attach-detach is enabled or not.. This call only valid for drivers which support attach & detach.

The following additional options are passed to the plugin as part of options map in FlexVolumeSource.
```
“kubernetes.io/host”=”<host-name>”
```

###### Request:
```
FlexVolumeAttachment
```

###### Response:
On success:
```
0 exit code
```
On failure:
```
non-zero exit code & errors.StatusError
```

###### Call out:
```
<driver executable> detach <json encoded FlexVolumeAttachment>
```

###### Call out sync: (Wait for detach)
```
<driver executable> detach <json encoded FlexVolumeAttachment> sync
```

#### Mount a volume:
This call mounts a volume on the node. It is executed from Kubelet. This is only valid for plugins which do not support attach/detach to a node. Example: NFS/CIFS. For volumes which support attach/detach, this will use the default logic to bind mount.

The following additional options are passed to the plugin as part of options map in FlexVolumeSource.
```
“kubernetes.io/mountpath”=”<mount path>”
```

###### New API Types:
```
type FlexVolumeMount struct {
    unversioned.TypeMeta
    v1.ObjectMeta
    MountPath string
}
```

###### Request:
```
FlexVolume
```

###### Response:
On success:
```
FlexVolumeMount
```
On failure:
```
non-zero exit code & errors.StatusError
```

###### Call out:
```
<driver executable> mount <json encoded FlexVolume>
```

#### Unmount a volume:
This call unmounts a volume on the node. It is executed from Kubelet. This is only valid for plugins which do not support attach/detach to a node. Example: NFS/CIFS.

###### Request:
```
FlexVolumeMount
```

###### Response:
On success:
```
0 exit code
```
On failure:
```
non-zero exit code & errors.StatusError
```

###### Call out:
```
<driver executable> unmount <json encoded FlexVolumeMount>
```

#### Metrics call
This call gets the metrics of a volume. It is executed from Kubelet.

###### New API Types:
```
type FlexVolumeMetrics struct {
    unversioned.TypeMeta
    v1.ObjectMeta
    Capacity *resource.Quantity
    Used *resource.Quantity
    Available *resource.Quantity
}
```

###### Request:
```
string
```

###### Response:
On success:
```
FlexVolumeMetrics
```
On failure:
```
non-zero exit code & errors.StatusError
```

###### Call out:
```
<driver executable> metrics <json encoded FlexVolumeMount>
```

## Dynamic Provisioning support:
The new driver probe function has support to query the capabilities and if the driver has of ‘dynamic provisioning’ capability, Flex volume plugin will support dynamic provisioning by implementing the ‘Provisioner’ and ‘Deleter’ interfaces.

## Attach detach support:
The new driver probe function has support to query the capabilities and if the driver has of ‘attachment’ capability, Flex volume plugin will support dynamic provisioning by implementing the ‘Provisioner’ and ‘Deleter’ interfaces.
PR by @MikaelCluseau (#26926) would be leveraged for this change.
