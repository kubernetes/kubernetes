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
###EnvVar###

---
* name: 
  * **_type_**: string
  * **_description_**: name of the environment variable; must be a C_IDENTIFIER
* value: 
  * **_type_**: string
  * **_description_**: value of the environment variable; defaults to empty string; variable references $(VAR_NAME) are expanded using the previously defined environment varibles in the container and any service environment variables; if a variable cannot be resolved, the reference in the input string will be unchanged; the $(VAR_NAME) syntax can be escaped with a double $$, ie: $$(VAR_NAME) ; escaped references will never be expanded, regardless of whether the variable exists or not
* valueFrom: 
  * **_type_**: [EnvVarSource](EnvVarSource.md)
  * **_description_**: source for the environment variable's value; cannot be used if value is not empty


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/EnvVar.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
