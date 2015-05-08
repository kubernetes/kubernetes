# Volumes
This document describes the current state of Volumes in kubernetes.  Familiarity with [pods](./pods.md) is suggested.

A Volume is a directory, possibly with some data in it, which is accessible to a Container. Kubernetes Volumes are similar to but not the same as [Docker Volumes](https://docs.docker.com/userguide/dockervolumes/).

A Pod specifies which Volumes its containers need in its [ContainerManifest](https://developers.google.com/compute/docs/containers/container_vms#container_manifest) property.

A process in a Container sees a filesystem view composed from two sources: a single Docker image and zero or more Volumes.  A [Docker image](https://docs.docker.com/userguide/dockerimages/) is at the root of the file hierarchy.  Any Volumes are mounted at points on the Docker image;  Volumes do not mount on other Volumes and do not have hard links to other Volumes.  Each container in the Pod independently specifies where on its image to mount each Volume.  This is specified a VolumeMounts property.

## Resources

The storage media (Disk, SSD, or memory) of a volume is determined by the media of the filesystem holding the kubelet root dir (typically `/var/lib/kubelet`).
There is no limit on how much space an EmptyDir or PersistentDir volume can consume, and no isolation between containers or between pods.

In the future, we expect that a Volume will be able to request a certain amount of space using a [resource](./resources.md) specification,
and to select the type of media to use, for clusters that have several media types.

## Types of Volumes

Kubernetes currently supports multiple types of Volumes. The community welcomes additional contributions.

### EmptyDir

An EmptyDir volume is created when a Pod is bound to a Node.  It is initially empty, when the first Container command starts.  Containers in the same pod can all read and write the same files in the EmptyDir.  When a Pod is unbound, the data in the EmptyDir is deleted forever.

Some uses for an EmptyDir are:
  - scratch space, such as for a disk-based mergesort or checkpointing a long computation.
  - a directory that a content-manager container fills with data while a webserver container serves the data.

Currently, the user cannot control what kind of media is used for an EmptyDir.  If the Kubelet is configured to use a disk drive, then all EmptyDirectories will be created on that disk drive.  In the future, it is expected that Pods can control whether the EmptyDir is on a disk drive, SSD, or tmpfs.

### HostDir
A Volume with a HostDir property allows access to files on the current node.

Some uses for a HostDir are:
  - running a container that needs access to Docker internals; use a HostDir of /var/lib/docker.
  - running cAdvisor in a container; use a HostDir of /dev/cgroups.

Watch out when using this type of volume, because:
  - pods with identical configuration (such as created from a podTemplate) may behave differently on different nodes due to different files on different nodes.
  - When Kubernetes adds resource-aware scheduling, as is planned, it will not be able to account for resources used by a HostDir.

### GCEPersistentDisk
__Important: You must create a PD using ```gcloud``` or the GCE API before you can use it__

A Volume with a GCEPersistentDisk property allows access to files on a Google Compute Engine (GCE)
[Persistent Disk](http://cloud.google.com/compute/docs/disks).

There are some restrictions when using a GCEPersistentDisk:
  - the nodes (what the kubelet runs on) need to be GCE VMs
  - those VMs need to be in the same GCE project and zone as the PD
  - avoid creating multiple pods that use the same Volume if any mount it read/write.
    - if a pod P already mounts a volume read/write, and a second pod Q attempts to use the volume, regardless of if it tries to use it read-only or read/write, Q will fail.
    - if a pod P already mounts a volume read-only, and a second pod Q attempts to use the volume read/write, Q will fail.
    - replication controllers with replicas > 1 can only be created for pods that use read-only mounts.

#### Creating a PD
Before you can use a GCE PD with a pod, you need to create it.

```sh
gcloud compute disks create --size=500GB --zone=us-central1-a my-data-disk
```

#### GCE PD Example configuration:
```yaml
apiVersion: v1beta1
desiredState:
  manifest:
    containers:
      - image: kubernetes/pause
        name: testpd
        volumeMounts:
          - mountPath: "/testpd"
            name: "testpd"
    id: testpd
    version: v1beta1
    volumes:
      - name: testpd
        source:
          persistentDisk:
            # This GCE PD must already exist.
            pdName: test
            fsType: ext4
id: testpd
kind: Pod
```
### NFS

Kubernetes NFS volumes allow an existing NFS share to be made available to containers within a pod.

See the [NFS Pod examples](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/nfs/) section for more details.
For example, [nfs-web-pod.yaml](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/nfs/nfs-web-pod.yaml) demonstrates how to specify the usage of an NFS volume within a pod.
In this example one can see that a `volumeMount` called "nfs" is being mounted onto `/var/www/html` in the container "web".
The volume "nfs" is defined as type `nfs`, with the NFS server serving from `nfs-server.default.kube.local` and exporting directory `/` as the share.
The mount being created in this example is not read only.
