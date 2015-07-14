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
