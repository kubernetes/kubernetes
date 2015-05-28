# Kubernetes v1 

Updated May 28, 2015

This document is intended to capture the set of supported use cases, features,
docs, and patterns that we feel are required to call Kubernetes “feature
complete” for a 1.0 release candidate. 

This list does not emphasize the bug fixes and stabilization that will be required to take it all the way to
production ready. Please see the [Github issues] (https://github.com/GoogleCloudPlatform/kubernetes/issues) for a more detailed view. 

This is a living document, where suggested changes can be made via a pull request.

## Target workloads

Most realistic examples of production services include a load-balanced web
frontend exposed to the public Internet, with a stateful backend, such as a
clustered database or key-value store. We will target such workloads for our
1.0 release.

## v1 APIs 
For existing and future workloads, we want to provide a consistent, stable set of APIs, over which developers can build and extend Kubernetes. This includes input validation, a consistent API structure, clean semantics, and improved diagnosability of the system. 

In addition, we will provide versioning and deprecation policies for the APIs.

## Cluster Environment
Currently, a cluster is a set of nodes (VMs, machines), managed by a master, running a version of Kubernetes. This master is the cluster-level control-plane. For the purpose of running production workloads, members of the cluster must be serviceable and upgradeable.

## Micro-services and Resources
For applications / micro-services that run on Kubernetes, we want deployments to be easy but powerful. An Operations user should be able to launch a micro-service, letting the scheduler find the right placement. That micro-service should be able to require “pet storage” resources, fulfilled by external storage and with help from the cluster. We also want to improve the tools, experience for how users can roll-out applications through patterns like canary deployments. 

## Performance and Reliability
The system should be performant, especially from the perspective of micro-service running on top of the cluster and for Operations users. As part of being production grade, the system should have a measured availability and be resilient to failures, including fatal failures due to hardware. 

In terms of performance, the objectives include:
- API call return times at 99%tile ([#4521](https://github.com/GoogleCloudPlatform/kubernetes/issues/4521))
- scale to 100 nodes with 30-50 pods (1-2 containers) per node
- scheduling throughput at the 99%tile ([#3954](https://github.com/GoogleCloudPlatform/kubernetes/issues/3954))
- startup time at the 99%tile ([#3552](https://github.com/GoogleCloudPlatform/kubernetes/issues/3952))


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/roadmap.md?pixel)]()
