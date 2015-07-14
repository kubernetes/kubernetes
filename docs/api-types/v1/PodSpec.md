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
###PodSpec###

---
* activeDeadlineSeconds: 
  * **_type_**: integer
  * **_description_**: 
* containers: 
  * **_type_**: [][Container](Container.md)
  * **_description_**: list of containers belonging to the pod; cannot be updated; containers cannot currently be added or removed; there must be at least one container in a Pod; see http://releases.k8s.io/HEAD/docs/containers.md
* dnsPolicy: 
  * **_type_**: string
  * **_description_**: DNS policy for containers within the pod; one of 'ClusterFirst' or 'Default'
* hostNetwork: 
  * **_type_**: boolean
  * **_description_**: host networking requested for this pod
* imagePullSecrets: 
  * **_type_**: [][LocalObjectReference](LocalObjectReference.md)
  * **_description_**: list of references to secrets in the same namespace available for pulling the container images; see http://releases.k8s.io/HEAD/docs/images.md#specifying-imagepullsecrets-on-a-pod
* nodeName: 
  * **_type_**: string
  * **_description_**: node requested for this pod
* nodeSelector: 
  * **_type_**: any
  * **_description_**: selector which must match a node's labels for the pod to be scheduled on that node; see http://releases.k8s.io/HEAD/examples/node-selection/README.md
* restartPolicy: 
  * **_type_**: string
  * **_description_**: restart policy for all containers within the pod; one of Always, OnFailure, Never; defaults to Always; see http://releases.k8s.io/HEAD/docs/pod-states.md#restartpolicy
* serviceAccountName: 
  * **_type_**: string
  * **_description_**: name of the ServiceAccount to use to run this pod; see http://releases.k8s.io/HEAD/docs/service_accounts.md
* terminationGracePeriodSeconds: 
  * **_type_**: integer
  * **_description_**: optional duration in seconds the pod needs to terminate gracefully; may be decreased in delete request; value must be non-negative integer; the value zero indicates delete immediately; if this value is not set, the default grace period will be used instead; the grace period is the duration in seconds after the processes running in the pod are sent a termination signal and the time when the processes are forcibly halted with a kill signal; set this value longer than the expected cleanup time for your process
* volumes: 
  * **_type_**: [][Volume](Volume.md)
  * **_description_**: list of volumes that can be mounted by containers belonging to the pod; see http://releases.k8s.io/HEAD/docs/volumes.md


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/PodSpec.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
