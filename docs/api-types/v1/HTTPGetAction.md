###HTTPGetAction###

---
* host: 
  * **_type_**: string
  * **_description_**: hostname to connect to; defaults to pod IP
* path: 
  * **_type_**: string
  * **_description_**: path to access on the HTTP server
* port: 
  * **_type_**: string
  * **_description_**: number or name of the port to access on the container; number must be in the range 1 to 65535; name must be an IANA_SVC_NAME
* scheme: 
  * **_type_**: string
  * **_description_**: scheme to connect with, must be HTTP or HTTPS, defaults to HTTP
