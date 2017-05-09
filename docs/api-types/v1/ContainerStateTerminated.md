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
###ContainerStateTerminated###

---
* containerID: 
  * **_type_**: string
  * **_description_**: container's ID in the format 'docker://<container_id>'
* exitCode: 
  * **_type_**: integer
  * **_description_**: exit status from the last termination of the container
* finishedAt: 
  * **_type_**: string
  * **_description_**: time at which the container last terminated
* message: 
  * **_type_**: string
  * **_description_**: message regarding the last termination of the container
* reason: 
  * **_type_**: string
  * **_description_**: (brief) reason from the last termination of the container
* signal: 
  * **_type_**: integer
  * **_description_**: signal from the last termination of the container
* startedAt: 
  * **_type_**: string
  * **_description_**: time at which previous execution of the container started


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/ContainerStateTerminated.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
