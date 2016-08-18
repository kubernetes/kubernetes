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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Flexvolume

Flexvolume enables users to support vendor volumes into kubernetes. It expects vendor drivers are installed in the volume plugin path on every Kubelet node.

It allows vendors to develop their own drivers and expose their volumes to Kubernetes pods.

*Note: Flexvolume is an alpha feature and is most likely to change in future*

## Prerequisites

Install the vendor driver on all nodes (including master nodes) in the Kubelet plugin path. Path for installing the plugin: /usr/libexec/kubernetes/kubelet-plugins/volume/exec/\<vendor~driver\>/\<driver\>

For example to add a 'cifs' driver, by vendor 'foo' install the driver at: /usr/libexec/kubernetes/kubelet-plugins/volume/exec/\<foo~cifs\>/cifs

## Plugin details

The plugin expects the following call-outs are implemented for all the backend drivers. Most of the call-outs are invoked from the Kubelet node.

### Driver invocation model:

#### Init:
Initializes the plugin.

```
<driver executable> init
```

#### Get volume name:
Query the flex volume driver for a globally unique name for the volume. This name will be used to uniquely identify the volume in Kubernetes cluster and to synchronize attaches and detaches between Controller-Manager & Kubelet.

```
<driver executable> getvolumename <json options>
```

#### Attach:
Attaches the volume specified by the given spec to the given host. On success, returns the device path where the device was attached on the node.

```
<driver executable> attach <json options> <hostname>
```

#### Wait for Attach:
Wait for the volume to be attached on the remote node. On success, the path to the device is returned.

```
<driver executable> waitforattach <json options>
```

#### Get Device Mount path:
GetDeviceMountPath returns a path where the device is mounted after it is attached. This is a global mount path which individual pods can then bind mount.
An extra option "kubernetes.io/mountsDir" is used to specify recommended mount path to mount the volume. Plugin can choose to override it and mount the volume at a different path and return it in "Status.Path". This call-out is optional.

```
<driver executable> getdevicemountpath <json options>
```

#### Mount device:
Mount device mounts the device to a global path which individual pods can then bind mount. This call-out is optional.

```
<driver executable> mountdevice <json options> <mount device> <mount dir>
```

#### Detach:
Detach the volume from the Kubelet node.

```
<driver executable> detach <mount device> <hostname>
```

#### Wait for Detach:
Wait for the volume to be detached from the Kubelet node.

```
<driver executable> waitfordetach <json options>
```

#### Unmount device:
Unmounts the global mount for the device. This is called once all bind mounts have been unmounted. This call-out is optional.

```
<driver executable> unmountdevice <mount device>
```

#### Mount:
Mounts the device at the mount path. This is called to bind mount globally mounted volume at the mount path. This call-out is optional.

```
<driver executable> mount  <json options> <mount device> <mount path>
```

#### Unmount:
Unmounts the device. This is called to unmount the bind mount. This call is optional.

```
<driver executable> unmount <mount path>
```

See [lvm](lvm) for a quick example on how to write a simple flexvolume driver. This is an example driver and it just imitates the options. It does not actually do any centralized attach/detach calls.

### Driver output:

Flexvolume expects the driver to reply with the status of the operation in the
following format.

```
{
	"status": "<Success/Failure>",
	"message": "<Reason for success/failure>",
	"device": "<Path to the device attached. This field is valid only for attach calls>",
	"path" : "<The path where the device is mounted>"
}
```

### Default Json options

In addition to the flags specified by the user in the Options field of the FlexVolumeSource, the following flags are also passed to the executable.

```
"kubernetes.io/fsType":"<FS type>",
"kubernetes.io/readwrite":"<rw>",
"kubernetes.io/fsGroup":"<FS group>",
"kubernetes.io/secret/key1":"<secret1>"
...
"kubernetes.io/secret/keyN":"<secretN>"
```

### Example of Flexvolume

See [nginx.yaml](nginx.yaml) for a quick example on how to use Flexvolume in a pod.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/flexvolume/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
