###ContainerPort###

---
* containerPort: 
  * **_type_**: integer
  * **_description_**: number of port to expose on the pod's IP address
* hostIP: 
  * **_type_**: string
  * **_description_**: host IP to bind the port to
* hostPort: 
  * **_type_**: integer
  * **_description_**: number of port to expose on the host; most containers do not need this
* name: 
  * **_type_**: string
  * **_description_**: name for the port that can be referred to by services; must be an IANA_SVC_NAME and unique within the pod
* protocol: 
  * **_type_**: string
  * **_description_**: protocol for port; must be UDP or TCP; TCP if unspecified
