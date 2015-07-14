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
###StatusDetails###

---
* causes: 
  * **_type_**: [][StatusCause](StatusCause.md)
  * **_description_**: the Causes array includes more details associated with the StatusReason failure; not all StatusReasons may provide detailed causes
* kind: 
  * **_type_**: string
  * **_description_**: the kind attribute of the resource associated with the status StatusReason; on some operations may differ from the requested resource Kind; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* name: 
  * **_type_**: string
  * **_description_**: the name attribute of the resource associated with the status StatusReason (when there is a single name which can be described)
* retryAfterSeconds: 
  * **_type_**: integer
  * **_description_**: the number of seconds before the client should attempt to retry this operation


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/StatusDetails.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
