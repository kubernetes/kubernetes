<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/examples/kubectl-container/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
To access the Kubernetes API [from a Pod](../../docs/user-guide/accessing-the-cluster.md#accessing-the-api-from-a-pod) one of the solution is to run `kubectl proxy` in a so-called sidecar container within the Pod. To do this, you need to package `kubectl` in a container. It is useful when service accounts are being used for accessing the API and the old no-auth KUBERNETES_RO service is not available. Since all containers in a Pod share the same network namespace, containers will be able to reach the API on localhost.

This example contains a [Dockerfile](Dockerfile) and [Makefile](Makefile) for packaging up `kubectl` into
a container and pushing the resulting container image on the Google Container Registry. You can modify the Makefile to push to a different registry if needed.

Assuming that you have checked out the Kubernetes source code and setup your environment to be able to build it. The typical build step of this kubectl container will be:

    $ cd examples/kubectl-container
    $ make kubectl
    $ make tag
    $ make container
    $ make push

It is not currently automated as part of a release process, so for the moment
this is an example of what to do if you want to package `kubectl` into a
container and use it within a pod.

In the future, we may release consistently versioned groups of containers when
we cut a release, in which case the source of gcr.io/google_containers/kubectl
would become that automated process.

[```pod.json```](pod.json) is provided as an example of running `kubectl` as a sidecar
container in a Pod, and to help you verify that `kubectl` works correctly in
this configuration. To launch this Pod, you will need a configured Kubernetes endpoint and `kubectl` installed locally, then simply create the Pod:

    $ kubectl create -f pod.json


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/kubectl-container/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
