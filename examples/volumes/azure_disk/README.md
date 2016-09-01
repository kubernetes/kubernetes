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
[here](http://releases.k8s.io/release-1.4/examples/volumes/azure_disk/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# How to Use it?

On Azure VM, create a Pod using the volume spec based on [azure](azure.yaml).

In the pod, you need to provide the following information:

- *diskName*:  (required) the name of the VHD blob object.
- *diskURI*: (required) the URI of the vhd blob object.
- *cachingMode*: (optional) disk caching mode. Must be one of None, ReadOnly, or ReadWrite. Default is None.
- *fsType*:  (optional) the filesytem type to mount. Default is ext4.
- *readOnly*: (optional) whether the filesystem is used as readOnly. Default is false.


Launch the Pod:

```console
    # kubectl create -f examples/volumes/azure_disk/azure.yaml
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/azure_disk/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
