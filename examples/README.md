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
[here](http://releases.k8s.io/release-1.3/examples/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes Examples: releases.k8s.io/HEAD

This directory contains a number of examples of how to run
real applications with Kubernetes.

Demonstrations of how to use specific Kubernetes features can be found in our [documents](../docs/).


### Maintained Examples

Maintained Examples are expected to be updated with every Kubernetes
release, to use the latest and greatest features, current guidelines
and best practices, and to refresh command syntax, output, changed
prerequisites, as needed.

Name | Description | Notable Features Used | Complexity Level
------------- | ------------- | ------------ | ------------ | ------------
[Guestbook](guestbook/) | PHP app with Redis | Replication Controller, Service | Beginner
[WordPress](mysql-wordpress-pd/) | WordPress with MySQL | Deployment, Persistent Volume with Claim | Beginner
[Cassandra](storage/cassandra/) | Cloud Native Cassandra | Daemon Set | Intermediate

Note: Please add examples to the list above that are maintained.

See [Example Guidelines](guidelines.md) for a description of what goes
in this directory, and what examples should contain.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
