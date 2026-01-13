# Package: testing

## Purpose
Provides test utilities and fake implementations for testing the attach/detach controller and related components.

## Key Functions

### Volume Spec Helpers
- **GetTestVolumeSpec(volumeName, diskName)**: Returns a test volume.Spec with GCE PD source.

### Fake Client
- **CreateTestClient(logger)**: Creates a fake Kubernetes client with pre-configured reactors for:
  - Pods (list/create with 5 default pods)
  - Nodes (list/update with 5 default nodes)
  - CSINodes (list with migrated plugin annotations)
  - VolumeAttachments (list/create)
  - PersistentVolumes (list/create)

### Object Factories
- **NewPod(uid, name)**: Creates a minimal test pod.
- **NewPodWithVolume(podName, volumeName, nodeName)**: Creates a pod with a GCE PD volume.
- **NewVolumeAttachment(vaName, pvName, nodeName, status)**: Creates a VolumeAttachment object.
- **NewPV(pvName, volumeName)**: Creates a PV with GCE PD source.
- **NewNFSPV(pvName, volumeName)**: Creates a PV with NFS source (non-migrated in-tree plugin).

### Internal Helpers
- **attachVolumeToNode**: Adds a volume to node's VolumesAttached status.

## Design Notes

- Test pods use "mynamespace" namespace and "mynode" as default node.
- Test volumes use GCE Persistent Disk by default (supports CSI migration testing).
- Includes "lostVolumeName" attached to test nodes for orphan volume testing.
- CSINode annotations indicate all in-tree plugins are migrated (v1.27+).
