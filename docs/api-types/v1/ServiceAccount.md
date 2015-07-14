###ServiceAccount###

---
* apiVersion: 
  * **_type_**: string
  * **_description_**: version of the schema the object should have; see http://releases.k8s.io/HEAD/docs/api-conventions.md#resources
* imagePullSecrets: 
  * **_type_**: [][LocalObjectReference](LocalObjectReference.md)
  * **_description_**: list of references to secrets in the same namespace available for pulling container images; see http://releases.k8s.io/HEAD/docs/secrets.md#manually-specifying-an-imagepullsecret
* kind: 
  * **_type_**: string
  * **_description_**: kind of object, in CamelCase; cannot be updated; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* metadata: 
  * **_type_**: [ObjectMeta](ObjectMeta.md)
  * **_description_**: standard object metadata; see http://releases.k8s.io/HEAD/docs/api-conventions.md#metadata
* secrets: 
  * **_type_**: [][ObjectReference](ObjectReference.md)
  * **_description_**: list of secrets that can be used by pods running as this service account; see http://releases.k8s.io/HEAD/docs/secrets.md
