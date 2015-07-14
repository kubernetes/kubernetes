###DeleteOptions###

---
* apiVersion: 
  * **_type_**: string
  * **_description_**: version of the schema the object should have; see http://releases.k8s.io/HEAD/docs/api-conventions.md#resources
* gracePeriodSeconds: 
  * **_type_**: integer
  * **_description_**: the duration in seconds to wait before deleting this object; defaults to a per object value if not specified; zero means delete immediately
* kind: 
  * **_type_**: string
  * **_description_**: kind of object, in CamelCase; cannot be updated; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
