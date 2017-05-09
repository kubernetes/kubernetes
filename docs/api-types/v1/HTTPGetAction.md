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


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/HTTPGetAction.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
