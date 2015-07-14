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
###PersistentVolumeStatus###

---
* message: 
  * **_type_**: string
  * **_description_**: human-readable message indicating details about why the volume is in this state
* phase: 
  * **_type_**: string
  * **_description_**: the current phase of a persistent volume; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#phase
* reason: 
  * **_type_**: string
  * **_description_**: (brief) reason the volume is not is not available


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/PersistentVolumeStatus.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
