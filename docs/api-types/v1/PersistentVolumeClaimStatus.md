###PersistentVolumeClaimStatus###

---
* accessModes: 
  * **_type_**: [][PersistentVolumeAccessMode](PersistentVolumeAccessMode.md)
  * **_description_**: the actual access modes the volume has; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#access-modes-1
* capacity: 
  * **_type_**: any
  * **_description_**: the actual resources the volume has
* phase: 
  * **_type_**: string
  * **_description_**: the current phase of the claim
