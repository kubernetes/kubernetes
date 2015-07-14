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
###SecurityContext###

---
* capabilities: 
  * **_type_**: [Capabilities](Capabilities.md)
  * **_description_**: the linux capabilites that should be added or removed; see http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context
* privileged: 
  * **_type_**: boolean
  * **_description_**: run the container in privileged mode; see http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context
* runAsUser: 
  * **_type_**: integer
  * **_description_**: the user id that runs the first process in the container; see http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context
* seLinuxOptions: 
  * **_type_**: [SELinuxOptions](SELinuxOptions.md)
  * **_description_**: options that control the SELinux labels applied; see http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/SecurityContext.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
