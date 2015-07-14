###ContainerStatus###

---
* containerID: 
  * **_type_**: string
  * **_description_**: container's ID in the format 'docker://<container_id>'; see http://releases.k8s.io/HEAD/docs/container-environment.md#container-information
* image: 
  * **_type_**: string
  * **_description_**: image of the container; see http://releases.k8s.io/HEAD/docs/images.md
* imageID: 
  * **_type_**: string
  * **_description_**: ID of the container's image
* lastState: 
  * **_type_**: [ContainerState](ContainerState.md)
  * **_description_**: details about the container's last termination condition
* name: 
  * **_type_**: string
  * **_description_**: name of the container; must be a DNS_LABEL and unique within the pod; cannot be updated
* ready: 
  * **_type_**: boolean
  * **_description_**: specifies whether the container has passed its readiness probe
* restartCount: 
  * **_type_**: integer
  * **_description_**: the number of times the container has been restarted, currently based on the number of dead containers that have not yet been removed
* state: 
  * **_type_**: [ContainerState](ContainerState.md)
  * **_description_**: details about the container's current condition
