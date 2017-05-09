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
###StatusCause###

---
* field: 
  * **_type_**: string
  * **_description_**: field of the resource that has caused this error, as named by its JSON serialization; may include dot and postfix notation for nested attributes; arrays are zero-indexed; fields may appear more than once in an array of causes due to fields having multiple errors
* message: 
  * **_type_**: string
  * **_description_**: human-readable description of the cause of the error; this field may be presented as-is to a reader
* reason: 
  * **_type_**: string
  * **_description_**: machine-readable description of the cause of the error; if this value is empty there is no information available


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/StatusCause.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
