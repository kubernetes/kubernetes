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
###PodStatus###

---
* conditions: 
  * **_type_**: [][PodCondition](PodCondition.md)
  * **_description_**: current service state of pod; see http://releases.k8s.io/HEAD/docs/pod-states.md#pod-conditions
* containerStatuses: 
  * **_type_**: [][ContainerStatus](ContainerStatus.md)
  * **_description_**: list of container statuses; see http://releases.k8s.io/HEAD/docs/pod-states.md#container-statuses
* hostIP: 
  * **_type_**: string
  * **_description_**: IP address of the host to which the pod is assigned; empty if not yet scheduled
* message: 
  * **_type_**: string
  * **_description_**: human readable message indicating details about why the pod is in this condition
* phase: 
  * **_type_**: string
  * **_description_**: current condition of the pod; see http://releases.k8s.io/HEAD/docs/pod-states.md#pod-phase
* podIP: 
  * **_type_**: string
  * **_description_**: IP address allocated to the pod; routable at least within the cluster; empty if not yet allocated
* reason: 
  * **_type_**: string
  * **_description_**: (brief-CamelCase) reason indicating details about why the pod is in this condition
* startTime: 
  * **_type_**: string
  * **_description_**: RFC 3339 date and time at which the object was acknowledged by the Kubelet.  This is before the Kubelet pulled the container image(s) for the pod.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/PodStatus.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
