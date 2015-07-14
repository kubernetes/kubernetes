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
###RBDVolumeSource###

---
* fsType: 
  * **_type_**: string
  * **_description_**: file system type to mount, such as ext4, xfs, ntfs; see http://releases.k8s.io/HEAD/examples/rbd/README.md#how-to-use-it
* image: 
  * **_type_**: string
  * **_description_**: rados image name; see http://releases.k8s.io/HEAD/examples/rbd/README.md#how-to-use-it
* keyring: 
  * **_type_**: string
  * **_description_**: keyring is the path to key ring for rados user; default is /etc/ceph/keyring; optional; see http://releases.k8s.io/HEAD/examples/rbd/README.md#how-to-use-it
* monitors: 
  * **_type_**: []string
  * **_description_**: a collection of Ceph monitors; see http://releases.k8s.io/HEAD/examples/rbd/README.md#how-to-use-it
* pool: 
  * **_type_**: string
  * **_description_**: rados pool name; default is rbd; optional; see http://releases.k8s.io/HEAD/examples/rbd/README.md#how-to-use-it
* readOnly: 
  * **_type_**: boolean
  * **_description_**: rbd volume to be mounted with read-only permissions; see http://releases.k8s.io/HEAD/examples/rbd/README.md#how-to-use-it
* secretRef: 
  * **_type_**: [LocalObjectReference](LocalObjectReference.md)
  * **_description_**: name of a secret to authenticate the RBD user; if provided overrides keyring; optional; see http://releases.k8s.io/HEAD/examples/rbd/README.md#how-to-use-it
* user: 
  * **_type_**: string
  * **_description_**: rados user name; default is admin; optional; see http://releases.k8s.io/HEAD/examples/rbd/README.md#how-to-use-it


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/RBDVolumeSource.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
