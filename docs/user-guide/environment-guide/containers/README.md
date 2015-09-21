<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/user-guide/environment-guide/containers/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Building
--------
For each container, the build steps are the same. The examples below
are for the `show` container. Replace `show` with `backend` for the
backend container.

Google Container Registry ([GCR](https://cloud.google.com/tools/container-registry/))
---
    docker build -t gcr.io/<project-name>/show .
    gcloud docker push gcr.io/<project-name>/show

Docker Hub
----------
    docker build -t <username>/show .
    docker push <username>/show

Change Pod Definitions
----------------------
Edit both `show-rc.yaml` and `backend-rc.yaml` and replace the
specified `image:` with the one that you built.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/environment-guide/containers/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
