<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Namespaces

Namespaces help different projects, teams, or customers to share a kubernetes cluster.  First, they provide a scope for [Names](../user-guide/identifiers.md).  Second, as our access control code develops, it is expected that it will be convenient to attach authorization and other policy to namespaces.

Use of multiple namespaces is optional.  For small teams, they may not be needed.

This is a placeholder document about namespace administration.

TODO: document namespace creation, ownership assignment, visibility rules,
policy creation, interaction with network.

Namespaces are still under development.  For now, the best documentation is the [Namespaces Design Document](../design/namespaces.md).  The user documentation can be found at [Namespaces](../../docs/user-guide/namespaces.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/admin/namespaces.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
