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
[here](http://releases.k8s.io/release-1.3/docs/proposals/volume-ownership-management.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Volume plugins and idempotency

Currently, volume plugins have a `SetUp` method which is called in the context of a higher-level
workflow within the kubelet which has externalized the problem of managing the ownership of volumes.
This design has a number of drawbacks that can be mitigated by completely internalizing all concerns
of volume setup behind the volume plugin `SetUp` method.

### Known issues with current externalized design

1.  The ownership management is currently repeatedly applied, which breaks packages that require
    special permissions in order to work correctly
2.  There is a gap between files being mounted/created by volume plugins and when their ownership
    is set correctly; race conditions exist around this
3.  Solving the correct application of ownership management in an externalized model is difficult
    and makes it clear that the a transaction boundary is being broken by the externalized design

### Additional issues with externalization

Fully externalizing any one concern of volumes is difficult for a number of reasons:

1.  Many types of idempotence checks exist, and are used in a variety of combinations and orders
2.  Workflow in the kubelet becomes much more complex to handle:
    1.  composition of plugins
    2.  correct timing of application of ownership management
    3.  callback to volume plugins when we know the whole `SetUp` flow is complete and correct
    4.  callback to touch sentinel files
    5.  etc etc
3.  We want to support fully external volume plugins -- would require complex orchestration / chatty
    remote API

## Proposed implementation

Since all of the ownership information is known in advance of the call to the volume plugin `SetUp`
method, we can easily internalize these concerns into the volume plugins and pass the ownership
information to `SetUp`.

The volume `Builder` interface's `SetUp` method changes to accept the group that should own the
volume.  Plugins become responsible for ensuring that the correct group is applied.  The volume
`Attributes` struct can be modified to remove the `SupportsOwnershipManagement` field.

```go
package volume

type Builder interface {
    // other methods omitted

    // SetUp prepares and mounts/unpacks the volume to a self-determined
    // directory path and returns an error.  The group ID that should own the volume
    // is passed as a parameter.  Plugins may choose to ignore the group ID directive
    // in the event that they do not support it (example: NFS).  A group ID of -1
    // indicates that the group ownership of the volume should not be modified by the plugin.
    //
    // SetUp will be called multiple times and should be idempotent.
    SetUp(gid int64) error
}
```

Each volume plugin will have to change to support the new `SetUp` signature.  The existing
ownership management code will be refactored into a library that volume plugins can use:

```
package volume

func ManageOwnership(path string, fsGroup int64) error {
    // 1. recursive chown of path
    // 2. make path +setgid
}
```

The workflow from the Kubelet's perspective for handling volume setup and refresh becomes:

```go
// go-ish pseudocode
func mountExternalVolumes(pod) error {
    podVolumes := make(kubecontainer.VolumeMap)
    for i := range pod.Spec.Volumes {
        volSpec := &pod.Spec.Volumes[i]
        var fsGroup int64 = 0
        if pod.Spec.SecurityContext != nil &&
            pod.Spec.SecurityContext.FSGroup != nil {
            fsGroup = *pod.Spec.SecurityContext.FSGroup
        } else {
            fsGroup = -1
        }

        // Try to use a plugin for this volume.
        plugin := volume.NewSpecFromVolume(volSpec)
        builder, err := kl.newVolumeBuilderFromPlugins(plugin, pod)
        if err != nil {
            return err
        }
        if builder == nil {
            return errUnsupportedVolumeType
        }

        err := builder.SetUp(fsGroup)
        if err != nil {
            return nil
        }
    }

    return nil
}
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/volume-ownership-management.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
