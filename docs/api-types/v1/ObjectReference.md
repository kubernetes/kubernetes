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
###ObjectReference###

---
* apiVersion: 
  * **_type_**: string
  * **_description_**: API version of the referent
* fieldPath: 
  * **_type_**: string
  * **_description_**: if referring to a piece of an object instead of an entire object, this string should contain a valid JSON/Go field access statement, such as desiredState.manifest.containers[2]
* kind: 
  * **_type_**: string
  * **_description_**: kind of the referent; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* name: 
  * **_type_**: string
  * **_description_**: name of the referent; see http://releases.k8s.io/HEAD/docs/identifiers.md#names
* namespace: 
  * **_type_**: string
  * **_description_**: namespace of the referent; see http://releases.k8s.io/HEAD/docs/namespaces.md
* resourceVersion: 
  * **_type_**: string
  * **_description_**: specific resourceVersion to which this reference is made, if any: http://releases.k8s.io/HEAD/docs/api-conventions.md#concurrency-control-and-consistency
* uid: 
  * **_type_**: string
  * **_description_**: uid of the referent; see http://releases.k8s.io/HEAD/docs/identifiers.md#uids


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/ObjectReference.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
