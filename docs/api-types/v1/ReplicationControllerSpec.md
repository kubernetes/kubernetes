###ReplicationControllerSpec###

---
* replicas: 
  * **_type_**: integer
  * **_description_**: number of replicas desired; defaults to 1; see http://releases.k8s.io/HEAD/docs/replication-controller.md#what-is-a-replication-controller
* selector: 
  * **_type_**: any
  * **_description_**: label keys and values that must match in order to be controlled by this replication controller, if empty defaulted to labels on Pod template; see http://releases.k8s.io/HEAD/docs/labels.md#label-selectors
* template: 
  * **_type_**: [PodTemplateSpec](PodTemplateSpec.md)
  * **_description_**: object that describes the pod that will be created if insufficient replicas are detected; takes precendence over templateRef; see http://releases.k8s.io/HEAD/docs/replication-controller.md#pod-template
