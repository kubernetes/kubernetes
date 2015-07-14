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
###NodeCondition###

---
* lastHeartbeatTime: 
  * **_type_**: string
  * **_description_**: last time we got an update on a given condition
* lastTransitionTime: 
  * **_type_**: string
  * **_description_**: last time the condition transit from one status to another
* message: 
  * **_type_**: string
  * **_description_**: human readable message indicating details about last transition
* reason: 
  * **_type_**: string
  * **_description_**: (brief) reason for the condition's last transition
* status: 
  * **_type_**: string
  * **_description_**: status of the condition, one of True, False, Unknown
* type: 
  * **_type_**: string
  * **_description_**: type of node condition, currently only Ready


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/NodeCondition.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
