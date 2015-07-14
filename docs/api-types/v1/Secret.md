###Secret###

---
* apiVersion: 
  * **_type_**: string
  * **_description_**: version of the schema the object should have; see http://releases.k8s.io/HEAD/docs/api-conventions.md#resources
* data: 
  * **_type_**: any
  * **_description_**: data contains the secret data.  Each key must be a valid DNS_SUBDOMAIN or leading dot followed by valid DNS_SUBDOMAIN.  Each value must be a base64 encoded string as described in https://tools.ietf.org/html/rfc4648#section-4
* kind: 
  * **_type_**: string
  * **_description_**: kind of object, in CamelCase; cannot be updated; see http://releases.k8s.io/HEAD/docs/api-conventions.md#types-kinds
* metadata: 
  * **_type_**: [ObjectMeta](ObjectMeta.md)
  * **_description_**: standard object metadata; see http://releases.k8s.io/HEAD/docs/api-conventions.md#metadata
* type: 
  * **_type_**: string
  * **_description_**: type facilitates programmatic handling of secret data
