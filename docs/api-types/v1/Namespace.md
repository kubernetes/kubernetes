###Namespace###

---
* apiVersion: 
  * **_type_**: string
  * **_description_**: version of the schema the object should have; see http://releases.k8s.io/HEAD/docs/api-conventions.md#resources
* kind: 
  * **_type_**: string
  * **_description_**: kind of object, in CamelCase; cannot be updated; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* metadata: 
  * **_type_**: [ObjectMeta](ObjectMeta.md)
  * **_description_**: standard object metadata; see http://releases.k8s.io/HEAD/docs/api-conventions.md#metadata
* spec: 
  * **_type_**: [NamespaceSpec](NamespaceSpec.md)
  * **_description_**: spec defines the behavior of the Namespace; http://releases.k8s.io/HEAD/docs/api-conventions.md#spec-and-status
* status: 
  * **_type_**: [NamespaceStatus](NamespaceStatus.md)
  * **_description_**: status describes the current status of a Namespace; http://releases.k8s.io/HEAD/docs/api-conventions.md#spec-and-status
