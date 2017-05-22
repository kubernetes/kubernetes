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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Hello world job

This example creates a hello world job that gives you some basic ideas of how to use Kubernetes.
It creates a job to greet you and exit.

You need [kubectl](http://kubernetes.io/docs/user-guide/prereqs/) with version 1.4 or higher to use flag `--quiet` and a [running kubernetes cluster](http://kubernetes.io/docs/getting-started-guides/) for this to work.

```sh
$ kubectl run hello-world --quiet --rm -i --image=busybox --restart=OnFailure -- echo "Hello from Kubernetes!"
Hello from Kubernetes!
```

For more information on how to run your application in a kubernetes cluster, please see the [user-guide](http://kubernetes.io/docs/user-guide/).



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/hello-world/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
