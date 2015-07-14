###Node###

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
  * **_type_**: [NodeSpec](NodeSpec.md)
  * **_description_**: specification of a node; http://releases.k8s.io/HEAD/docs/api-conventions.md#spec-and-status
* status: 
  * **_type_**: [NodeStatus](NodeStatus.md)
  * **_description_**: most recently observed status of the node; populated by the system, read-only; http://releases.k8s.io/HEAD/docs/api-conventions.md#spec-and-status
