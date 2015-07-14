###ReplicationControllerList###

---
* apiVersion: 
  * **_type_**: string
  * **_description_**: version of the schema the object should have; see http://releases.k8s.io/HEAD/docs/api-conventions.md#resources
* items: 
  * **_type_**: [][ReplicationController](ReplicationController.md)
  * **_description_**: list of replication controllers; see http://releases.k8s.io/HEAD/docs/replication-controller.md
* kind: 
  * **_type_**: string
  * **_description_**: kind of object, in CamelCase; cannot be updated; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* metadata: 
  * **_type_**: [ListMeta](ListMeta.md)
  * **_description_**: standard list metadata; see http://releases.k8s.io/HEAD/docs/api-conventions.md#metadata
