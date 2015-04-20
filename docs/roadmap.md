# Kubernetes Roadmap

Updated April 20, 2015

This document is intended to capture the set of supported use cases, features,
docs, and patterns that we feel are required to call Kubernetes “feature
complete” for a 1.0 release candidate.  This list does not emphasize the bug
fixes and stabilization that will be required to take it all the way to
production ready.  This is a living document, and is certainly open for
discussion.

## Target workloads

Most realistic examples of production services include a load-balanced web
frontend exposed to the public Internet, with a stateful backend, such as a
clustered database or key-value store. We will target such workloads for our
1.0 release.

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

## Project
1. Define a deprecation policy for expiring and removing features and interfaces, including the time non-beta APIs will be supported
  - Status:
2. Define a version numbering policy regarding point upgrades, support, compat, and release frequency.
  - Status:
3. Define an SLO that users can reasonable expect to hit in properly managed clusters
  - Status:
4. Accurate and complete API documentation
  - Status:
5. Accurate and complete getting-started-guides for supported platforms
  - Status:

## Platforms
1. Possible for cloud partners / vendors to self-qualify Kubernetes on their platform.
  - Status:
2. Define the set of platforms that are supported by the core team.
  - Status:

## Beyond 1.0

We acknowledge that there are a great many things that are not included in our 1.0 roadmap.  We intend to document the plans past 1.0 soon, but some of the things that are clearly in scope include:

1. Scalability - more nodes, more pods
2. HA masters
3. Monitoring
4. Authn and authz
5. Enhanced resource management and isolation
6. Better performance
7. Easier plugins and add-ons
8. More support for jobs that complete (compute, batch)
9. More platforms
10. Easier testing
