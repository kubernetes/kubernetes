# Kubernetes Developer Guide

The developer guide is for anyone wanting to either write code which directly accesses the
kubernetes API, or to contribute directly to the kubernetes project.
It assumes some familiarity with concepts in the [User Guide](http://releases.k8s.io/HEAD/docs/user-guide.md) and the [Cluster Admin
Guide](cluster-admin-guide.md).


## Developing against the Kubernetes API

* API objects are explained at [http://kubernetes.io/third_party/swagger-ui/](http://kubernetes.io/third_party/swagger-ui/).

* **Annotations** ([annotations.md](http://releases.k8s.io/HEAD/docs/annotations.md)): are for attaching arbitrary non-identifying metadata to objects.
  Programs that automate Kubernetes objects may use annotations to store small amounts of their state.

* **API Conventions** ([api-conventions.md](http://releases.k8s.io/HEAD/docs/api-conventions.md)):
  Defining the verbs and resources used in the Kubernetes API.

* **API Client Libraries** ([client-libraries.md](http://releases.k8s.io/HEAD/docs/client-libraries.md)):
  A list of existing client libraries, both supported and user-contributed.

## Writing Plugins

* **Authentication Plugins** ([authentication.md](http://releases.k8s.io/HEAD/docs/authentication.md)):
  The current and planned states of authentication tokens.

* **Authorization Plugins** ([authorization.md](http://releases.k8s.io/HEAD/docs/authorization.md)):
  Authorization applies to all HTTP requests on the main apiserver port.
  This doc explains the available authorization implementations.

* **Admission Control Plugins** ([admission_control](http://releases.k8s.io/HEAD/docs/design/admission_control.md))

## Contributing to the Kubernetes Project

See this [README](http://releases.k8s.io/HEAD/docs/../docs/devel/README.md).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/developer-guide.md?pixel)]()
