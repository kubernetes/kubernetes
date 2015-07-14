###ISCSIVolumeSource###

---
* fsType: 
  * **_type_**: string
  * **_description_**: file system type to mount, such as ext4, xfs, ntfs
* iqn: 
  * **_type_**: string
  * **_description_**: iSCSI Qualified Name
* lun: 
  * **_type_**: integer
  * **_description_**: iscsi target lun number
* readOnly: 
  * **_type_**: boolean
  * **_description_**: read-only if true, read-write otherwise (false or unspecified)
* targetPortal: 
  * **_type_**: string
  * **_description_**: iSCSI target portal
