## App Container basics

[App Container][appc-repo] (appc) is an open specification that defines several aspects of how to run applications in containers: an image format, runtime environment, and discovery protocol.

rkt's native [image format](#aci) and [runtime environment](#pods) are those defined by the [specification][appc-spec].

## ACI

The image format defined by appc and used in rkt is the [_Application Container Image_][appc-aci], or ACI.
An ACI is a simple tarball bundle of a rootfs (containing all the files needed to execute an application) and an _Image Manifest_, which defines things like default execution parameters and default resource constraints.
ACIs can be built with tools like [`acbuild`][acbuild], [`actool`][actool], or [`goaci`][goaci].
Docker images can be converted to ACI using [`docker2aci`][docker2aci], although rkt will [do this automatically][running-docker-images].

Most parameters defined in an image can be overridden at runtime by rkt. For example, the `rkt run` command allows users to supply custom exec arguments to an image.

## Pods

appc defines the [_pod_][appc-pods] as the basic unit of execution.
A pod is a grouping of one or more app images (ACIs), with some additional metadata optionally applied to the pod as a whole - for example, a resource constraint can be applied at the pod level and then forms an "outer bound" for all the applications in the pod.
The images in a pod execute with a shared context, including networking.

A pod in rkt is conceptually identical to a pod [as defined in Kubernetes][k8s-pods].

## Validating rkt as an appc implementation

rkt implements the two runtime components of the appc specification: the [Application Container Executor (ACE)][appc-ace] and the [Metadata Service][appc-meta].
It also uses schema and code from the upstream [appc/spec][appc-spec] repo to manipulate ACIs, work with image and pod manifests, and perform image discovery.

To validate that `rkt` successfully implements the ACE part of the spec, use the App Container [validation ACIs][appc-val]:

```
# rkt metadata-service &  # Make sure metadata service is running
# rkt --insecure-options=image run \
	--mds-register \
	--volume=database,kind=host,source=/tmp \
	https://github.com/appc/spec/releases/download/v0.8.10/ace-validator-main.aci \
	https://github.com/appc/spec/releases/download/v0.8.10/ace-validator-sidekick.aci
```

[acbuild]: https://github.com/containers/build
[actool]: https://github.com/appc/spec#building-acis
[appc-repo]: https://github.com/appc/spec/
[appc-spec]: https://github.com/appc/spec/blob/master/SPEC.md
[appc-aci]: https://github.com/appc/spec/blob/master/spec/aci.md#app-container-image
[appc-pods]: https://github.com/appc/spec/blob/master/spec/pods.md#app-container-pods-pods
[appc-ace]: https://github.com/appc/spec/blob/master/spec/ace.md#app-container-executor
[appc-meta]: https://github.com/appc/spec/blob/master/spec/ace.md#app-container-metadata-service
[appc-val]: https://github.com/appc/spec/blob/master/README.md#validating-app-container-executors-aces
[docker2aci]: https://github.com/appc/docker2aci
[goaci]: https://github.com/appc/goaci
[k8s-pods]: http://kubernetes.io/docs/user-guide/pods/
[running-docker-images]: running-docker-images.md
