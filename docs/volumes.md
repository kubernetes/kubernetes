# Volumes
This document describes the current state of Volumes in kubernetes.  Familiarity with [pods](./pods.md) is suggested.


A Volume is a directory, possibly with some data in it, which is accessible to a Container. Kubernetes Volumes are similar to but not the same as [Docker Volumes](https://docs.docker.com/userguide/dockervolumes/).

A Pod specifies which Volumes its containers need in its [ContainerManifest](https://developers.google.com/compute/docs/containers/container_vms#container_manifest) property.

A process in a Container sees a filesystem view composed from two sources: a single Docker image and zero or more Volumes.  A [Docker image](https://docs.docker.com/userguide/dockerimages/) is at the root of the file hierarchy.  Any Volumes are mounted at points on the Docker image;  Volumes do not mount on other Volumes and do not have hard links to other Volumes.  Each container in the Pod independently specifies where on its its image to mount each Volume.  This is specified a VolumeMount property.

## Types of Volumes

Kubernetes currently supports two types of Volumes, but more may be added in the future.

### EmptyDirectory

An EmptyDirectory volume is created when a Pod is bound to a Node.  It is initially empty, when the first Container command starts.  Containers in the same pod can all read and write the same files in the EmptyDirectory.  When a Pod is unbound, the data in the EmptyDirectory is deleted forever.

Some uses for an EmptyDirectory are:
  - scratch space, such as for a disk-based mergesort or checkpointing a long computation.
  - a directory that a content-manager container fills with data while a webserver container serves the data.

Currently, the user cannot control what kind of media is used for an EmptyDirectory.  If the Kubelet is configured to use a disk drive, then all EmptyDirectories will be created on that disk drive.  In the future, it is expected that Pods can control whether the EmptyDirectory is on a disk drive, SSD, or tmpfs.

### HostDirectory
A Volume with a HostDirectory property allows access to files on the current node.

Some uses for a HostDirectory are:
  - running a container that need access to Docker internals; use a HostDirectory of /var/lib/docker.
  - running cAdvisor in a container; use a HostDirectory of /dev/cgroups.

Watch out when using this type of volume, because:
  - pods with identical configuration (such as created from a PodTemplate) may behave differently on different nodes, if they see node-specific files.
  - When Kubernetes adds resource-aware scheduling, as is planned, it will not be able to account for resources used by a HostDirectory.

