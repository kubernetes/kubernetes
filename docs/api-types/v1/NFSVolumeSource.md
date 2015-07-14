###NFSVolumeSource###

---
* path: 
  * **_type_**: string
  * **_description_**: the path that is exported by the NFS server; see http://releases.k8s.io/HEAD/docs/volumes.md#nfs
* readOnly: 
  * **_type_**: boolean
  * **_description_**: forces the NFS export to be mounted with read-only permissions; see http://releases.k8s.io/HEAD/docs/volumes.md#nfs
* server: 
  * **_type_**: string
  * **_description_**: the hostname or IP address of the NFS server; see http://releases.k8s.io/HEAD/docs/volumes.md#nfs
