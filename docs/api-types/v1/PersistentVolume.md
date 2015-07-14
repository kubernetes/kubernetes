###PersistentVolume###

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
  * **_type_**: [PersistentVolumeSpec](PersistentVolumeSpec.md)
  * **_description_**: specification of a persistent volume as provisioned by an administrator; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#persistent-volumes
* status: 
  * **_type_**: [PersistentVolumeStatus](PersistentVolumeStatus.md)
  * **_description_**: current status of a persistent volume; populated by the system, read-only; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#persistent-volumes
