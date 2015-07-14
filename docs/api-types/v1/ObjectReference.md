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
