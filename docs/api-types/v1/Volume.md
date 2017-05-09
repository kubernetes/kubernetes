<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
###Volume###

---
* awsElasticBlockStore: 
  * **_type_**: [AWSElasticBlockStoreVolumeSource](AWSElasticBlockStoreVolumeSource.md)
  * **_description_**: AWS disk resource attached to the host machine on demand; see http://releases.k8s.io/HEAD/docs/volumes.md#awselasticblockstore
* emptyDir: 
  * **_type_**: [EmptyDirVolumeSource](EmptyDirVolumeSource.md)
  * **_description_**: temporary directory that shares a pod's lifetime; see http://releases.k8s.io/HEAD/docs/volumes.md#emptydir
* gcePersistentDisk: 
  * **_type_**: [GCEPersistentDiskVolumeSource](GCEPersistentDiskVolumeSource.md)
  * **_description_**: GCE disk resource attached to the host machine on demand; see http://releases.k8s.io/HEAD/docs/volumes.md#gcepersistentdisk
* gitRepo: 
  * **_type_**: [GitRepoVolumeSource](GitRepoVolumeSource.md)
  * **_description_**: git repository at a particular revision
* glusterfs: 
  * **_type_**: [GlusterfsVolumeSource](GlusterfsVolumeSource.md)
  * **_description_**: Glusterfs volume that will be mounted on the host machine; see http://releases.k8s.io/HEAD/examples/glusterfs/README.md
* hostPath: 
  * **_type_**: [HostPathVolumeSource](HostPathVolumeSource.md)
  * **_description_**: pre-existing host file or directory; generally for privileged system daemons or other agents tied to the host; see http://releases.k8s.io/HEAD/docs/volumes.md#hostpath
* iscsi: 
  * **_type_**: [ISCSIVolumeSource](ISCSIVolumeSource.md)
  * **_description_**: iSCSI disk attached to host machine on demand; see http://releases.k8s.io/HEAD/examples/iscsi/README.md
* name: 
  * **_type_**: string
  * **_description_**: volume name; must be a DNS_LABEL and unique within the pod; see http://releases.k8s.io/HEAD/docs/identifiers.md#names
* nfs: 
  * **_type_**: [NFSVolumeSource](NFSVolumeSource.md)
  * **_description_**: NFS volume that will be mounted in the host machine; see http://releases.k8s.io/HEAD/docs/volumes.md#nfs
* persistentVolumeClaim: 
  * **_type_**: [PersistentVolumeClaimVolumeSource](PersistentVolumeClaimVolumeSource.md)
  * **_description_**: a reference to a PersistentVolumeClaim in the same namespace; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#persistentvolumeclaims
* rbd: 
  * **_type_**: [RBDVolumeSource](RBDVolumeSource.md)
  * **_description_**: rados block volume that will be mounted on the host machine; see http://releases.k8s.io/HEAD/examples/rbd/README.md
* secret: 
  * **_type_**: [SecretVolumeSource](SecretVolumeSource.md)
  * **_description_**: secret to populate volume; see http://releases.k8s.io/HEAD/docs/volumes.md#secrets


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/Volume.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
