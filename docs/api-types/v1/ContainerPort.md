<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
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


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/ContainerPort.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
