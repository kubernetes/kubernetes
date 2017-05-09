# Persistent Storage Provisioning

This document proposes a design taking advantage of cloud provider APIs to automate provisioning of persistent storage.


## Background

### Kubernetes Infrastructure (current and under development).

Kubernetes currently has a mechanism for users to
[request storage](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/design/persistent-storage.md#request-storage)
for their pods, and a mechanism for admins to
[describe the storage](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/design/persistent-storage.md#describe-available-storage)
they have created and made available for use by pods. The various
mechanisms for mounting the provided storage are provided by plugins
such as the
[GCE PD](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/pkg/kubelet/volume/gce_pd/gce_pd.go),
[NFS plugin](https://github.com/GoogleCloudPlatform/kubernetes/pull/4601),
the [iSCSI plugin](https://github.com/GoogleCloudPlatform/kubernetes/pull/4612)

For more detail please see the [Persistent Storage design document](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/design/persistent-storage.md)

### Cloud Provider APIs

Cloud providers usually provide an API for requesting storage dynamically.

For example:

GCE: https://cloud.google.com/compute/docs/reference/latest/disks/insert

AWS: http://docs.aws.amazon.com/AmazonS3/latest/API/SOAPCreateBucket.html

Rackspace: http://api.rackspace.com/api-ref-blockstorage.html#volumes


## Proposal

With all the building blocks mentioned above in place it is now possible to add functionality to use cloud APIs to
automatically provide this storage as needed. The goal of this proposal is to reduce the work required of administrator to
specifying a description of the persistent volumes needed. Kubernetes will then know which APIs to use to provision storage.

To do this I propose adding the following components:

**'PersistentVolumeProvider'**: A new interface which requires Create and Recycle methods.
The actual implementation of using the APIs to request the storage form the cloud provider and announce its availability will be
delegated to cloud specific plugins in a manner similar to how the persistent volume mounting is delegated to volume type
specific implementations.

**'PersistentVolumeController'**: A new API type analogous to a 'replicationControler'. A 'PersistentVolumeController' uses
a the 'PersistentVolumeProvier' to ensure that a specified number of persistent volumes is created and ready to be matched to
claims. As claims get matched the controller creates more volumes to ensure the requested level is maintained in the pool.
The controller also watches for claims deleted by users -marked 'Recycled'- and takes care of recycling and returning those
volumes to the pool.

It is worth mentioning that this does not have to be restricted to cloud storage providers. Any storage technology which has
the ability to enforce quota can be supported by a plugin. In that scenario an administrator creates a large pool of storage
and makes it available to the cluster. The administrator then creates a 'PersistentVolumeProvider' corresponding to the storage
protocol type and the particular plugin takes care of requesting and releasing storage as pods request it.
