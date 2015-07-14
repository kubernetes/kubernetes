<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes Developer Guide

The developer guide is for anyone wanting to either write code which directly accesses the
kubernetes API, or to contribute directly to the kubernetes project.
It assumes some familiarity with concepts in the [User Guide](../user-guide/user-guide.md) and the [Cluster Admin
Guide](../admin/README.md).


## Developing against the Kubernetes API

* API objects are explained at [http://kubernetes.io/third_party/swagger-ui/](http://kubernetes.io/third_party/swagger-ui/).

* **Annotations** ([docs/user-guide/annotations.md](../user-guide/annotations.md)): are for attaching arbitrary non-identifying metadata to objects.
  Programs that automate Kubernetes objects may use annotations to store small amounts of their state.

* **API Conventions** ([api-conventions.md](api-conventions.md)):
  Defining the verbs and resources used in the Kubernetes API.

* **API Client Libraries** ([client-libraries.md](client-libraries.md)):
  A list of existing client libraries, both supported and user-contributed.

## Writing Plugins

* **Authentication Plugins** ([docs/admin/authentication.md](../admin/authentication.md)):
  The current and planned states of authentication tokens.

* **Authorization Plugins** ([docs/admin/authorization.md](../admin/authorization.md)):
  Authorization applies to all HTTP requests on the main apiserver port.
  This doc explains the available authorization implementations.

* **Admission Control Plugins** ([admission_control](../design/admission_control.md))

## Contributing to the Kubernetes Project

See this [README](README.md).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/developer-guide.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
