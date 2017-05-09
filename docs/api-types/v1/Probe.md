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
###Probe###

---
* exec: 
  * **_type_**: [ExecAction](ExecAction.md)
  * **_description_**: exec-based handler
* httpGet: 
  * **_type_**: [HTTPGetAction](HTTPGetAction.md)
  * **_description_**: HTTP-based handler
* initialDelaySeconds: 
  * **_type_**: integer
  * **_description_**: number of seconds after the container has started before liveness probes are initiated; see http://releases.k8s.io/HEAD/docs/pod-states.md#container-probes
* tcpSocket: 
  * **_type_**: [TCPSocketAction](TCPSocketAction.md)
  * **_description_**: TCP-based handler; TCP hooks not yet supported
* timeoutSeconds: 
  * **_type_**: integer
  * **_description_**: number of seconds after which liveness probes timeout; defaults to 1 second; see http://releases.k8s.io/HEAD/docs/pod-states.md#container-probes


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/Probe.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
