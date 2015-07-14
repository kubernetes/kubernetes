###Lifecycle###

---
* postStart: 
  * **_type_**: [Handler](Handler.md)
  * **_description_**: called immediately after a container is started; if the handler fails, the container is terminated and restarted according to its restart policy; other management of the container blocks until the hook completes; see http://releases.k8s.io/HEAD/docs/container-environment.md#hook-details
* preStop: 
  * **_type_**: [Handler](Handler.md)
  * **_description_**: called before a container is terminated; the container is terminated after the handler completes; other management of the container blocks until the hook completes; see http://releases.k8s.io/HEAD/docs/container-environment.md#hook-details
