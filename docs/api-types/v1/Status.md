###Status###

---
* apiVersion: 
  * **_type_**: string
  * **_description_**: version of the schema the object should have; see http://releases.k8s.io/HEAD/docs/api-conventions.md#resources
* code: 
  * **_type_**: integer
  * **_description_**: suggested HTTP return code for this status; 0 if not set
* details: 
  * **_type_**: [StatusDetails](StatusDetails.md)
  * **_description_**: extended data associated with the reason; each reason may define its own extended details; this field is optional and the data returned is not guaranteed to conform to any schema except that defined by the reason type
* kind: 
  * **_type_**: string
  * **_description_**: kind of object, in CamelCase; cannot be updated; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* message: 
  * **_type_**: string
  * **_description_**: human-readable description of the status of this operation
* metadata: 
  * **_type_**: [ListMeta](ListMeta.md)
  * **_description_**: standard list metadata; see http://releases.k8s.io/HEAD/docs/api-conventions.md#metadata
* reason: 
  * **_type_**: string
  * **_description_**: machine-readable description of why this operation is in the 'Failure' status; if this value is empty there is no information available; a reason clarifies an HTTP status code but does not override it
* status: 
  * **_type_**: string
  * **_description_**: status of the operation; either Success, or Failure; see http://releases.k8s.io/HEAD/docs/api-conventions.md#spec-and-status
