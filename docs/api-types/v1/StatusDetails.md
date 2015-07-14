###StatusDetails###

---
* causes: 
  * **_type_**: [][StatusCause](StatusCause.md)
  * **_description_**: the Causes array includes more details associated with the StatusReason failure; not all StatusReasons may provide detailed causes
* kind: 
  * **_type_**: string
  * **_description_**: the kind attribute of the resource associated with the status StatusReason; on some operations may differ from the requested resource Kind; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* name: 
  * **_type_**: string
  * **_description_**: the name attribute of the resource associated with the status StatusReason (when there is a single name which can be described)
* retryAfterSeconds: 
  * **_type_**: integer
  * **_description_**: the number of seconds before the client should attempt to retry this operation
