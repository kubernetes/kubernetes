# Kubernetes Developer Guide

The developer guide is for anyone wanting to either write code which directly accesses the
kubernetes API, or to contribute directly to the kubernetes project.
It assumes some familiarity with concepts in the [User Guide](user-guide.md) and the [Cluster Admin
Guide](cluster-admin-guide.md).


## Developing against the Kubernetes API

* API objects are explained at [http://kubernetes.io/third_party/swagger-ui/](http://kubernetes.io/third_party/swagger-ui/).

* **Annotations** ([annotations.md](annotations.md)): are for attaching arbitrary non-identifying metadata to objects.
  Programs that automate Kubernetes objects may use annotations to store small amounts of their state.

* **API Conventions** ([api-conventions.md](api-conventions.md)):
  Defining the verbs and resources used in the Kubernetes API.

* **API Client Libraries** ([client-libraries.md](client-libraries.md)):
  A list of existing client libraries, both supported and user-contributed.

## Writing Plugins

* **Authentication Plugins** ([authentication.md](authentication.md)):
  The current and planned states of authentication tokens.

* **Authorization Plugins** ([authorization.md](authorization.md)):
  Authorization applies to all HTTP requests on the main apiserver port.
  This doc explains the available authorization implementations.

## Contributing to the Kubernetes Project

See this [README](../docs/devel/README.md).
