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
[here](http://releases.k8s.io/release-1.0/examples/k8petstore/bps-data-generator/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# How to generate the bps-data-generator container #

This container is maintained as part of the apache bigtop project.

To create it, simply

`git clone https://github.com/apache/bigtop`

and checkout the last exact version (will be updated periodically).

`git checkout -b aNewBranch 2b2392bf135e9f1256bd0b930f05ae5aef8bbdcb`

then, cd to bigtop-bigpetstore/bigpetstore-transaction-queue, and run the docker file, i.e.

`Docker build -t -i jayunit100/bps-transaction-queue`.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/k8petstore/bps-data-generator/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
