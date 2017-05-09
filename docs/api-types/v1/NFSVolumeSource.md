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


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/NFSVolumeSource.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
