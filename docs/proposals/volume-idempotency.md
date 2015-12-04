<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/volume-idempotency.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Volume plugins and idempotency

Currently, volume plugins have a `SetUp` method which is called repeatedly in the kubelet sync loop.
Implementers of plugins must implement idempotency checks in `SetUp` to avoid doing duplicate work.
In order to simplify life for developers of plugins and correctly support idempotent application
of ownership management to volumes, we should externalize these idempotency checks and perform them
in the kubelet.

### Types of idempotence checks

There are two types of idempotence checks that volumes perform today:

1.  Checking for the presence of a mount point
2.  Checking for the presence of a marker file for the volume in the metadata directory for the
    plugin type

Externalized idempotency management must support both of these types of checks.  The marker file
method requires that the marker file be created after `SetUp` is run successfully.

### Non-idempotent volumes

The hostPath volume does not implement an idempotency check because its `SetUp` method is a no-op.
It should be possible to represent that a volume does not have any idempotency concerns.

### Refreshable volume plugins

Some volume plugins are designed to refresh, like the downward API volume plugin.  These volume
plugins can be implemented by having a nop idempotence check; ie, setup always runs.  Ownership
management should be triggered when SetUp() does work for these plugins.

### Volume plugins that wrap one another

Some volume plugins wrap others; secrets, downward API, git repo are all based on the EmptyDir
plugin.  In order to correctly externalize idempotency management, the Kubelet must handle nested
calls to `SetUp`; plugins must be able to invoke the externalized flow correctly for plugins they
use transitively.

## Ownership management and idempotency

Ownership management should always happen when `SetUp` makes changes in volumes.  For volumes like
block devices, this means running once and only once after `Setup` has run correctly.  If a volume
that uses the 'check for a mountpoint' type of idempotency check, and the volume must be remounted,
the ownership management should be reapplied.

For volume plugins that are refreshable, ownership management must be reapplied if the refresh logic
makes changes to the state of the volume.  For example, if data changes in the downward API volume,
ownership management must be re-applied to that volume.

## Proposed implementation

Summarizing the requirements for correct idempotency management:

1.  It must be possible to express that a volume plugin has no idempotency check
2.  It must be possible to express that a volume plugin uses either a mountpoint check or marker
    file
3.  Marker file volumes must have the marker file created after `SetUp` runs successfully
4.  Ownership management must run once for each time `Setup` performs work
5.  Volume plugins must be able to trigger the externalized Setup flow to support correct
    composition of plugins

The volume `Attributes` structure is the appropriate place to store information the properties we
discuss here:

```go
package volume

type Attributes struct {
	// other fields omitted

    // IdempotencyStrategy controls the strategy used to manage idempotency for the volume.
	IdempotencyStrategy IdempotencyStrategy
}

type IdempotencyStrategyType string

const (
    IdempotencyStrategyNone       IdempotencyStrategyType = "none"
    IdempotencyStrategyMountpoint IdempotencyStrategyType = "mountPoint"
    IdempotencyStrategyMarkerFile IdempotencyStrategyType = "markerFile"
)
```

At runtime, these checks are represented by an interface:

```go
package volume

type IdempotencyStrategy interface {
    // ShouldRun returns (ok-to-run-setup, error)
    ShouldRun() (bool, error)
    Mark() error
}
```

The volume `Builder` interface's `SetUp` method changes signature to return `(bool, error)` to
indicate whether work was done.  Ownership management is performed for supported volumes whenever
`SetUp` does work.

```go
package volume

type Builder interface {
    // other methods omitted

    // SetUp prepares and mounts/unpacks the volume to a self-determined
    // directory path and returns whether work was done and an error.
    // SetUp will be called multiple times for volumes that select IdempotencyStrategyNone.
    SetUp() (bool, error)
}
```

The workflow from the Kubelet's perspective for handling volume setup and refresh is:

```go
// go-ish pseudocode
func mountExternalVolumes(pod) error {
    podVolumes := make(kubecontainer.VolumeMap)
    for i := range pod.Spec.Volumes {
        volSpec := &pod.Spec.Volumes[i]
        hasFSGroup := false
        var fsGroup int64 = 0
        if pod.Spec.SecurityContext != nil &&
            pod.Spec.SecurityContext.FSGroup != nil {
            hasFSGroup = true
            fsGroup = *pod.Spec.SecurityContext.FSGroup
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

        didWork, err := runSetup(builder)
        if err != nil {
            return nil
        }

        if didWork &&
            hasFSGroup &&
            builder.GetAttributes().Managed &&
            builder.GetAttributes().SupportsOwnershipManagement {
            err := manageVolumeOwnership(pod, plugin, builder, fsGroup)
            if err != nil {
                return err
            } 
        }  
    }

    return nil
}

// runSetup calls the SetUp method on a builder if idempotency strategy determines that it should
// be called and returns a boolean indicating whether work was done and an error.
func runSetup(builder) (bool, error) {
    strategyName := builder.GetAttributes().IdempotencyStrategyName
    if strategy == IdempotencyStrategyNone {
        idempotencyStrategy := makeIdempotencyStrategy(strategy)
        runSetup, err := idempotencyStrategy.ShouldRun()
        if err != nil {
            return false, err
        }

        if !runSetup {
            return false, nil
        }
    }
    
    didWork, err = builder.SetUp()
    if err != nil {
        return false, err
        idempotencyStrategy.Mark()
    }

    return didWork, nil
}
```

Volumes that require composition must be able to invoke the externalized workflow on a per-plugin
basis.  The logical extension point for this in the volume `Host` interface:

```go
package volume

type Host interface {
    // other methods omitted

    SetUp(builder Builder) (bool, error)
}
```

The implementation of this method at runtime will call the same `runSetup` flow used by the Kubelet.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/volume-idempotency.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
