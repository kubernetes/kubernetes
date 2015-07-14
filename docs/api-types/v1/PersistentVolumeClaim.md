###PersistentVolumeClaim###

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
  * **_type_**: [PersistentVolumeClaimSpec](PersistentVolumeClaimSpec.md)
  * **_description_**: the desired characteristics of a volume; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#persistentvolumeclaims
* status: 
  * **_type_**: [PersistentVolumeClaimStatus](PersistentVolumeClaimStatus.md)
  * **_description_**: the current status of a persistent volume claim; read-only; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#persistentvolumeclaims
