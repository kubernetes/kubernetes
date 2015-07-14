<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Building
--------
For each container, the build steps are the same. The examples below
are for the `show` container. Replace `show` with `backend` for the
backend container.

GCR
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
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/environment-guide/containers/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
