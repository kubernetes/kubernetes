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
###Container###

---
* args: 
  * **_type_**: []string
  * **_description_**: command array; the docker image's cmd is used if this is not provided; arguments to the entrypoint; cannot be updated; variable references $(VAR_NAME) are expanded using the container's environment variables; if a variable cannot be resolved, the reference in the input string will be unchanged; the $(VAR_NAME) syntax can be escaped with a double $$, ie: $$(VAR_NAME) ; escaped references will never be expanded, regardless of whether the variable exists or not; see http://releases.k8s.io/HEAD/docs/containers.md#containers-and-commands
* command: 
  * **_type_**: []string
  * **_description_**: entrypoint array; not executed within a shell; the docker image's entrypoint is used if this is not provided; cannot be updated; variable references $(VAR_NAME) are expanded using the container's environment variables; if a variable cannot be resolved, the reference in the input string will be unchanged; the $(VAR_NAME) syntax can be escaped with a double $$, ie: $$(VAR_NAME) ; escaped references will never be expanded, regardless of whether the variable exists or not; see http://releases.k8s.io/HEAD/docs/containers.md#containers-and-commands
* env: 
  * **_type_**: [][EnvVar](EnvVar.md)
  * **_description_**: list of environment variables to set in the container; cannot be updated
* image: 
  * **_type_**: string
  * **_description_**: Docker image name; see http://releases.k8s.io/HEAD/docs/images.md
* imagePullPolicy: 
  * **_type_**: string
  * **_description_**: image pull policy; one of Always, Never, IfNotPresent; defaults to Always if :latest tag is specified, or IfNotPresent otherwise; cannot be updated; see http://releases.k8s.io/HEAD/docs/images.md#updating-images
* lifecycle: 
  * **_type_**: [Lifecycle](Lifecycle.md)
  * **_description_**: actions that the management system should take in response to container lifecycle events; cannot be updated
* livenessProbe: 
  * **_type_**: [Probe](Probe.md)
  * **_description_**: periodic probe of container liveness; container will be restarted if the probe fails; cannot be updated; see http://releases.k8s.io/HEAD/docs/pod-states.md#container-probes
* name: 
  * **_type_**: string
  * **_description_**: name of the container; must be a DNS_LABEL and unique within the pod; cannot be updated
* ports: 
  * **_type_**: [][ContainerPort](ContainerPort.md)
  * **_description_**: list of ports to expose from the container; cannot be updated
* readinessProbe: 
  * **_type_**: [Probe](Probe.md)
  * **_description_**: periodic probe of container service readiness; container will be removed from service endpoints if the probe fails; cannot be updated; see http://releases.k8s.io/HEAD/docs/pod-states.md#container-probes
* resources: 
  * **_type_**: [ResourceRequirements](ResourceRequirements.md)
  * **_description_**: Compute Resources required by this container; cannot be updated; see http://releases.k8s.io/HEAD/docs/compute_resources.md
* securityContext: 
  * **_type_**: [SecurityContext](SecurityContext.md)
  * **_description_**: security options the pod should run with; see http://releases.k8s.io/HEAD/docs/security_context.md
* terminationMessagePath: 
  * **_type_**: string
  * **_description_**: path at which the file to which the container's termination message will be written is mounted into the container's filesystem; message written is intended to be brief final status, such as an assertion failure message; defaults to /dev/termination-log; cannot be updated
* volumeMounts: 
  * **_type_**: [][VolumeMount](VolumeMount.md)
  * **_description_**: pod volumes to mount into the container's filesyste; cannot be updated
* workingDir: 
  * **_type_**: string
  * **_description_**: container's working directory; defaults to image's default; cannot be updated


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/api-types/v1/Container.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
