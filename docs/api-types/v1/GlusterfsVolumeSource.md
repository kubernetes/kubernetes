###GlusterfsVolumeSource###

---
* endpoints: 
  * **_type_**: string
  * **_description_**: gluster hosts endpoints name; see http://releases.k8s.io/HEAD/examples/glusterfs/README.md#create-a-pod
* path: 
  * **_type_**: string
  * **_description_**: path to gluster volume; see http://releases.k8s.io/HEAD/examples/glusterfs/README.md#create-a-pod
* readOnly: 
  * **_type_**: boolean
  * **_description_**: glusterfs volume to be mounted with read-only permissions; see http://releases.k8s.io/HEAD/examples/glusterfs/README.md#create-a-pod
