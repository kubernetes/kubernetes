<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


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



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/azure_disk/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
