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
###ReplicationControllerSpec###

---
* replicas: 
  * **_type_**: integer
  * **_description_**: number of replicas desired; defaults to 1; see http://releases.k8s.io/HEAD/docs/replication-controller.md#what-is-a-replication-controller
* selector: 
  * **_type_**: any
  * **_description_**: label keys and values that must match in order to be controlled by this replication controller, if empty defaulted to labels on Pod template; see http://releases.k8s.io/HEAD/docs/labels.md#label-selectors
* template: 
  * **_type_**: [PodTemplateSpec](PodTemplateSpec.md)
  * **_description_**: object that describes the pod that will be created if insufficient replicas are detected; takes precendence over templateRef; see http://releases.k8s.io/HEAD/docs/replication-controller.md#pod-template


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/ReplicationControllerSpec.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
