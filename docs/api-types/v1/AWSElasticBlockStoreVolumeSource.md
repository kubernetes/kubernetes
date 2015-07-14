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
###AWSElasticBlockStoreVolumeSource###

---
* fsType: 
  * **_type_**: string
  * **_description_**: file system type to mount, such as ext4, xfs, ntfs; see http://releases.k8s.io/HEAD/docs/volumes.md#awselasticblockstore
* partition: 
  * **_type_**: integer
  * **_description_**: partition on the disk to mount (e.g., '1' for /dev/sda1); if omitted the plain device name (e.g., /dev/sda) will be mounted; see http://releases.k8s.io/HEAD/docs/volumes.md#awselasticblockstore
* readOnly: 
  * **_type_**: boolean
  * **_description_**: read-only if true, read-write otherwise (false or unspecified); see http://releases.k8s.io/HEAD/docs/volumes.md#awselasticblockstore
* volumeID: 
  * **_type_**: string
  * **_description_**: unique id of the PD resource in AWS; see http://releases.k8s.io/HEAD/docs/volumes.md#awselasticblockstore


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/AWSElasticBlockStoreVolumeSource.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
