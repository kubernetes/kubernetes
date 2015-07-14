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
###Event###

---
* apiVersion: 
  * **_type_**: string
  * **_description_**: version of the schema the object should have; see http://releases.k8s.io/HEAD/docs/api-conventions.md#resources
* count: 
  * **_type_**: integer
  * **_description_**: the number of times this event has occurred
* firstTimestamp: 
  * **_type_**: string
  * **_description_**: the time at which the event was first recorded
* involvedObject: 
  * **_type_**: [ObjectReference](ObjectReference.md)
  * **_description_**: object this event is about
* kind: 
  * **_type_**: string
  * **_description_**: kind of object, in CamelCase; cannot be updated; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* lastTimestamp: 
  * **_type_**: string
  * **_description_**: the time at which the most recent occurance of this event was recorded
* message: 
  * **_type_**: string
  * **_description_**: human-readable description of the status of this operation
* metadata: 
  * **_type_**: [ObjectMeta](ObjectMeta.md)
  * **_description_**: standard object metadata; see http://releases.k8s.io/HEAD/docs/api-conventions.md#metadata
* reason: 
  * **_type_**: string
  * **_description_**: short, machine understandable string that gives the reason for the transition into the object's current status
* source: 
  * **_type_**: [EventSource](EventSource.md)
  * **_description_**: component reporting this event


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/Event.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
