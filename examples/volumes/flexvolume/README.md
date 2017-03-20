# Flexvolume

Flexvolume enables users to write their own drivers and add support for their volumes in Kubernetes. Vendor drivers should be installed in the volume plugin path on every Kubelet node and on master node(s) if "--enable-controller-attach-detach" Kubelet option is enabled. 

*Note: Flexvolume is an alpha feature and is most likely to change in future*

## Prerequisites

Install the vendor driver on all nodes (also on master nodes if "--enable-controller-attach-detach" Kubelet option is enabled) in the plugin path. Path for installing the plugin: /usr/libexec/kubernetes/kubelet-plugins/volume/exec/\<vendor~driver\>/\<driver\>

For example to add a 'cifs' driver, by vendor 'foo' install the driver at: /usr/libexec/kubernetes/kubelet-plugins/volume/exec/\<foo~cifs\>/cifs

## Plugin details
The plugin expects the following call-outs are implemented for the backend drivers. Some call-outs are optional. Call-outs are invoked from the Kubelet & the Controller manager nodes.
Call-outs are invoked from Controller-manager only when "--enable-controller-attach-detach" Kubelet option is enabled.

### Driver invocation model:

#### Init:
Initializes the driver. Called during Kubelet & Controller manager initialization.

```
<driver executable> init
```

#### Get volume name:
Get a cluster wide unique volume name for the volume. Called from both Kubelet & Controller manager.

```
<driver executable> getvolumename <json options>
```

#### Attach:
Attach the volume specified by the given spec on the given host. On success, returns the device path where the device is attached on the node. Nodename param is only valid/relevant if "--enable-controller-attach-detach" Kubelet option is enabled. Called from both Kubelet & Controller manager.

This call-out does not pass "secrets" specified in Flexvolume spec. If your driver requires secrets, do not implement this call-out and instead use "mount" call-out and implement attach and mount in that call-out.

```
<driver executable> attach <json options> <node name>
```

#### Detach:
Detach the volume from the Kubelet node. Nodename param is only valid/relevant if "--enable-controller-attach-detach" Kubelet option is enabled. Called from both Kubelet & Controller manager.
```
<driver executable> detach <mount device> <node name>
```

#### Wait for attach:
Wait for the volume to be attached on the remote node. On success, the path to the device is returned. Called from both Kubelet & Controller manager.

```
<driver executable> waitforattach <mount device> <json options>
```

#### Volume is Attached:
Check the volume is attached on the node. Called from both Kubelet & Controller manager.

```
<driver executable> isattached <json options> <node name>
```

#### Mount device:
Mount device mounts the device to a global path which individual pods can then bind mount. Called only from Kubelet.

This call-out does not pass "secrets" specified in Flexvolume spec. If your driver requires secrets, do not implement this call-out and instead use "mount" call-out and implement attach and mount in that call-out.

```
<driver executable> mountdevice <mount dir> <mount device> <json options>
```

#### Unmount device:
Unmounts the global mount for the device. This is called once all bind mounts have been unmounted. Called only from Kubelet.

```
<driver executable> unmountdevice <mount device>
```

#### Mount:
Mount the volume at the mount dir. This call-out defaults to bind mount for drivers which implement attach & mount-device call-outs. Called only from Kubelet.

```
<driver executable> mount <mount dir> <json options>
```

#### Unmount:
Unmount the volume. This call-out defaults to bind mount for drivers which implement attach & mount-device call-outs. Called only from Kubelet.

```
<driver executable> unmount <mount dir>
```

See [lvm](lvm) & [nfs](nfs) for a quick example on how to write a simple flexvolume driver.

### Driver output:

Flexvolume expects the driver to reply with the status of the operation in the
following format.

```
{
	"status": "<Success/Failure/Not supported>",
	"message": "<Reason for success/failure>",
	"device": "<Path to the device attached. This field is valid only for attach & waitforattach call-outs>"
	"volumeName": "<Cluster wide unique name of the volume. Valid only for getvolumename call-out>"
	"attached": <True/False (Return true if volume is attached on the node. Valid only for isattached call-out)>
}
```

### Default Json options

In addition to the flags specified by the user in the Options field of the FlexVolumeSource, the following flags are also passed to the executable.
Note: Secrets are passed only to "mount/unmount" call-outs.

```
"kubernetes.io/fsType":"<FS type>",
"kubernetes.io/readwrite":"<rw>",
"kubernetes.io/secret/key1":"<secret1>"
...
"kubernetes.io/secret/keyN":"<secretN>"
```

### Example of Flexvolume

See [nginx.yaml](nginx.yaml) & [nginx-nfs.yaml](nginx-nfs.yaml) for a quick example on how to use Flexvolume in a pod.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/flexvolume/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
