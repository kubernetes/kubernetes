###PersistentVolumeClaimSpec###

---
* accessModes: 
  * **_type_**: [][PersistentVolumeAccessMode](PersistentVolumeAccessMode.md)
  * **_description_**: the desired access modes the volume should have; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#access-modes-1
* resources: 
  * **_type_**: [ResourceRequirements](ResourceRequirements.md)
  * **_description_**: the desired resources the volume should have; see http://releases.k8s.io/HEAD/docs/persistent-volumes.md#resources
* volumeName: 
  * **_type_**: string
  * **_description_**: the binding reference to the persistent volume backing this claim
