# Kubernetes Developer Guide

The developer guide is for anyone wanting to either write code which directly accesses the
Kubernetes API, or to contribute directly to the Kubernetes project.
It assumes some familiarity with concepts in the [User Guide](../user-guide/README.md) and the [Cluster Admin
Guide](../admin/README.md).


## The process of developing and contributing code to the Kubernetes project

* **On Collaborative Development** ([collab.md](collab.md)): Info on pull requests and code reviews.

* **GitHub Issues** ([issues.md](issues.md)): How incoming issues are reviewed and prioritized.

* **Pull Request Process** ([pull-requests.md](pull-requests.md)): When and why pull requests are closed.

* **Kubernetes On-Call Rotations** ([on-call-rotations.md](on-call-rotations.md)): Descriptions of on-call rotations for build and end-user support.

* **Faster PR reviews** ([faster_reviews.md](faster_reviews.md)): How to get faster PR reviews.

* **Getting Recent Builds** ([getting-builds.md](getting-builds.md)): How to get recent builds including the latest builds that pass CI.

* **Automated Tools** ([automation.md](automation.md)): Descriptions of the automation that is running on our github repository.


## Setting up your dev environment, coding, and debugging

* **Development Guide** ([development.md](development.md)): Setting up your development environment.

* **Hunting flaky tests** ([flaky-tests.md](flaky-tests.md)): We have a goal of 99.9% flake free tests.
  Here's how to run your tests many times.

* **Logging Conventions** ([logging.md](logging.md)): Glog levels.

* **Profiling Kubernetes** ([profiling.md](profiling.md)): How to plug in go pprof profiler to Kubernetes.

* **Instrumenting Kubernetes with a new metric**
  ([instrumentation.md](instrumentation.md)): How to add a new metrics to the
  Kubernetes code base.

* **Coding Conventions** ([coding-conventions.md](coding-conventions.md)):
  Coding style advice for contributors.

* **Document Conventions** ([how-to-doc.md](how-to-doc.md))
  Document style advice for contributors.

* **Running a cluster locally** ([running-locally.md](running-locally.md)):
  A fast and lightweight local cluster deployment for development.

## Developing against the Kubernetes API

* The [REST API documentation](../api-reference/README.md) explains the REST
  API exposed by apiserver.

* **Annotations** ([docs/user-guide/annotations.md](../user-guide/annotations.md)): are for attaching arbitrary non-identifying metadata to objects.
  Programs that automate Kubernetes objects may use annotations to store small amounts of their state.

* **API Conventions** ([api-conventions.md](api-conventions.md)):
  Defining the verbs and resources used in the Kubernetes API.

* **API Client Libraries** ([client-libraries.md](client-libraries.md)):
  A list of existing client libraries, both supported and user-contributed.


## Writing plugins

* **Authentication Plugins** ([docs/admin/authentication.md](../admin/authentication.md)):
  The current and planned states of authentication tokens.

* **Authorization Plugins** ([docs/admin/authorization.md](../admin/authorization.md)):
  Authorization applies to all HTTP requests on the main apiserver port.
  This doc explains the available authorization implementations.

* **Admission Control Plugins** ([admission_control](../design/admission_control.md))


## Building releases

See the [kubernetes/release](https://github.com/kubernetes/release) repository for details on creating releases and related tools and helper scripts.

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
