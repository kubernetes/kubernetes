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
###PersistentVolumeClaimStatus###

---
* accessModes: 
  * **_type_**: [][PersistentVolumeAccessMode](PersistentVolumeAccessMode.md)
  * **_description_**: the actual access modes the volume has; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#access-modes-1
* capacity: 
  * **_type_**: any
  * **_description_**: the actual resources the volume has
* phase: 
  * **_type_**: string
  * **_description_**: the current phase of the claim


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/PersistentVolumeClaimStatus.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
