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
# NFS-exporter container

Inspired by https://github.com/cpuguy83/docker-nfs-server. Rewritten for
Fedora.

Serves NFS4 exports, defined on command line. At least one export must be defined!

Usage::

    docker run -d --name nfs --privileged jsafrane/nfsexporter /path/to/share /path/to/share2 ...


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/nfs/exporter/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
