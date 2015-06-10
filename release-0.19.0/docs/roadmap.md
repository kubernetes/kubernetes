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
||||||| merged common ancestors
## APIs and core features
1. Consistent v1 API
  - Status: DONE. [v1beta3](http://kubernetesio.blogspot.com/2015/04/introducing-kubernetes-v1beta3.html) was developed as the release candidate for the v1 API.
2. Multi-port services for apps which need more than one port on the same portal IP ([#1802](https://github.com/GoogleCloudPlatform/kubernetes/issues/1802))
  - Status: DONE. Released in 0.15.0
3. Nominal services for applications which need one stable IP per pod instance ([#260](https://github.com/GoogleCloudPlatform/kubernetes/issues/260))
  - Status: #2585 covers some design options.
4. API input is scrubbed of status fields in favor of a new API to set status ([#4248](https://github.com/GoogleCloudPlatform/kubernetes/issues/4248))
  - Status: DONE
5. Input validation reporting versioned field names ([#3084](https://github.com/GoogleCloudPlatform/kubernetes/issues/3084))
  - Status: in progress
6. Error reporting: Report common problems in ways that users can discover
  - Status:
7. Event management: Make events usable and useful
  - Status:
8. Persistent storage support ([#5105](https://github.com/GoogleCloudPlatform/kubernetes/issues/5105))
  - Status: in progress
9. Allow nodes to join/leave a cluster ([#6087](https://github.com/GoogleCloudPlatform/kubernetes/issues/6087),[#3168](https://github.com/GoogleCloudPlatform/kubernetes/issues/3168))
  - Status: in progress ([#6949](https://github.com/GoogleCloudPlatform/kubernetes/pull/6949))
10. Handle node death
  - Status: mostly covered by nodes joining/leaving a cluster
11. Allow live cluster upgrades ([#6075](https://github.com/GoogleCloudPlatform/kubernetes/issues/6075),[#6079](https://github.com/GoogleCloudPlatform/kubernetes/issues/6079))
  - Status: design in progress
12. Allow kernel upgrades
  - Status: mostly covered by nodes joining/leaving a cluster, need demonstration
13. Allow rolling-updates to fail gracefully ([#1353](https://github.com/GoogleCloudPlatform/kubernetes/issues/1353))
  - Status:
14. Easy .dockercfg
  - Status:
15. Demonstrate cluster stability over time
  - Status
16. Kubelet use the kubernetes API to fetch jobs to run (instead of etcd) on supported platforms
  - Status: DONE

## Reliability and performance

1. Restart system components in case of crash (#2884)
  - Status: in progress
2. Scale to 100 nodes (#3876)
  - Status: in progress
3. Scale to 30-50 pods (1-2 containers each) per node (#4188)
  - Status:
4. Scheduling throughput: 99% of scheduling decisions made in less than 1s on 100 node, 3000 pod cluster; linear time to number of nodes and pods (#3954)
5. Startup time: 99% of end-to-end pod startup time with prepulled images is less than 5s on 100 node, 3000 pod cluster; linear time to number of nodes and pods (#3952, #3954)
  - Status:
6. API performance: 99% of API calls return in less than 1s; constant time to number of nodes and pods (#4521)
  - Status:
7. Manage and report disk space on nodes (#4135)
  - Status: in progress
8. API test coverage more than 85% in e2e tests
  - Status:

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


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/roadmap.md?pixel)]()
