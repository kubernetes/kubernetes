###GCEPersistentDiskVolumeSource###

---
* fsType: 
  * **_type_**: string
  * **_description_**: file system type to mount, such as ext4, xfs, ntfs; see http://releases.k8s.io/HEAD/docs/volumes.md#gcepersistentdisk
* partition: 
  * **_type_**: integer
  * **_description_**: partition on the disk to mount (e.g., '1' for /dev/sda1); if omitted the plain device name (e.g., /dev/sda) will be mounted; see http://releases.k8s.io/HEAD/docs/volumes.md#gcepersistentdisk
* pdName: 
  * **_type_**: string
  * **_description_**: unique name of the PD resource in GCE; see http://releases.k8s.io/HEAD/docs/volumes.md#gcepersistentdisk
* readOnly: 
  * **_type_**: boolean
  * **_description_**: read-only if true, read-write otherwise (false or unspecified); see http://releases.k8s.io/HEAD/docs/volumes.md#gcepersistentdisk
