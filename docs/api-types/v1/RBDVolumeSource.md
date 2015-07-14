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
