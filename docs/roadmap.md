# Kubernetes Roadmap

Updated December 6, 2014

This document is intended to capture the set of supported use cases, features, docs, and patterns that we feel are required to call Kubernetes “feature complete” for a 1.0 release candidate.  This list does not emphasize the bug fixes and stabilization that will be required to take it all the way to production ready.  This is a living document, and is certainly open for discussion.

## Target workloads

Features for 1.0 will be driven by the initial set of workloads we intend to support.

Most realistic examples of production services include a load-balanced web frontend exposed to the public Internet, with a stateful backend, such as a clustered database or key-value store, so we will target such a workload for 1.0.

Which exact stateful applications are TBD. Candidates include:
* redis
* memcache
* mysql (using master/slave replication)
* mongo
* cassandra
* etcd
* zookeeper

## APIs
1. Consistent v1 API. [v1beta3](https://github.com/GoogleCloudPlatform/kubernetes/issues/1519) is being developed as the release candidate for the v1 API.
2. Deprecation policy: Declare the project’s intentions with regards to expiring and removing features and interfaces, including the minimum amount of time non-beta APIs will be supported.
3. Input validation: Validate schemas of API requests in the apiserver and, optionally, in the client.
4. Error propagation: Report problems reliably and consistently, with documented behavior.
5. Easy to add new controllers, such as [per-node controller](https://github.com/GoogleCloudPlatform/kubernetes/pull/2491)
  1. Replication controller: Make replication controller a standalone entity in the master stack.
  2. Pod templates: Proposal to make pod templates a first-class API object, rather than an artifact of replica controller [#170](https://github.com/GoogleCloudPlatform/kubernetes/issues/170)
6. Kubelet API should be well defined and versioned.
7. Cloud provider API for managing nodes, storage, and network resources. [#2770](https://github.com/GoogleCloudPlatform/kubernetes/issues/2770)

## Scheduling and resource isolation
1. Resource requirements and scheduling: Use knowledge of resources available and resources required to make good enough scheduling decisions such that applications can start and run. [#168](https://github.com/GoogleCloudPlatform/kubernetes/issues/168)

## Images and registry
1. Simple out-of-the box registry setup. [#1319](https://github.com/GoogleCloudPlatform/kubernetes/issues/1319)
2. Easy to configure .dockercfg.
3. Easy to deploy new code to Kubernetes (build and push).
4. Predictable deployment via configuration-time image resolution. [#1697](https://github.com/GoogleCloudPlatform/kubernetes/issues/1697)

## Storage
1. Durable volumes: Provide a model for data with identity and lifetime independent of pods. [#1515](https://github.com/GoogleCloudPlatform/kubernetes/pull/1515), [#598](https://github.com/GoogleCloudPlatform/kubernetes/issues/598), [#2609](https://github.com/GoogleCloudPlatform/kubernetes/pull/2609)
2. Pluggable volume sources and devices: Allow new kinds of data sources and/or devices as volumes. [#945](https://github.com/GoogleCloudPlatform/kubernetes/issues/945), [#2598](https://github.com/GoogleCloudPlatform/kubernetes/pull/2598)

## Networking and naming
1. DNS: Provide DNS for services, internal and external. [#2224](https://github.com/GoogleCloudPlatform/kubernetes/pull/2224), [#1261](https://github.com/GoogleCloudPlatform/kubernetes/issues/1261)
2. External IPs: Make Kubernetes services externally reachable. [#1161](https://github.com/GoogleCloudPlatform/kubernetes/issues/1161)
3. Re-think the network parts of the API: Clean factoring of a la carte networking functionality. [#2585](https://github.com/GoogleCloudPlatform/kubernetes/issues/2585)
4. Out-of-the-box, kick-the-tires networking implementation. [#1307](https://github.com/GoogleCloudPlatform/kubernetes/issues/1307)

## Authentication and authorization
1. Auth[nz] and ACLs: Have a plan for how the API and system will express:
  1. Identity & authentication
  2. Authorization & access control
  3. Cluster subdivision, accounting, & isolation
2. Support for pluggable authentication implementation and authorization polices
3. Implemented auth[nz] for:
   1. admin to master and/or kubelet
   2. user to master
   3. master component to component (e.g., controller manager to apiserver): localhost in 1.0
   4. kubelet to master

## Usability

### Documentation
1. Documnted reference cluster architecture
2. Accurate and complete API documentation

### Cluster turnup, scaling, management, and upgrades
1. Easy cluster startup
  1. Automatic node registration
  2. Configuring k8s
    1. Move away from flags in master
    2. Node configuration distribution
       1. Kubelet configuration
       2. dockercfg
2. Easy cluster scaling (adding/removing nodes)
3. Kubernetes can be upgraded
  1. master components
  2. Kubelets
  3. OS + kernel + Docker

### Workload deployment and management
1. Kubectl fully replaces kubecfg [#2144](https://github.com/GoogleCloudPlatform/kubernetes/issues/2144)
  1. Graceful termination. [#1535](https://github.com/GoogleCloudPlatform/kubernetes/issues/1535)
  2. Resize. [#1629](https://github.com/GoogleCloudPlatform/kubernetes/issues/1629)
  3. Config generators integrated into kubectl.
  4. Rolling updates. [#1353](https://github.com/GoogleCloudPlatform/kubernetes/issues/1353)
2. Kubectl can perform bulk operations (e.g., delete) on streams of API objects [#1905](https://github.com/GoogleCloudPlatform/kubernetes/issues/1905)

## Productionization
1. Scalability
  1. 100 nodes for 1.0
  2. 1000 nodes by summer 2015
2. HA master -- not gating 1.0
  1. Master election
  2. Eliminate global in-memory state
    1. IP allocator
    2. Operations
  3. Sharding
    1. Pod getter
3. Kubelets need to coast when master down
  1. Don’t blow away pods when master is down
4. Testing
  1. More/better/easier E2E
  2. E2E integration testing w/ OpenShift
  3. More non-E2E integration tests
  4. Long-term soaking / stress test
  5. Backward compatibility
    1. API
    2. etcd state
5. Release cadence and artifacts
  1. Regular stable releases on a frequent timeline (2 weeks).
  2. Automatic generation of necessary deployment artifacts. It is still TBD if this includes deb and RPMs. Also not clear if it includes docker containers.
6. Export monitoring metrics (instrumentation)
7. Bounded disk space on master and kubelets
  1. GC of unused images

### Performance Improvement Work Items
We are currently collating pain points and work items for improving the performance of Kubernetes.
Please contribute material to the Google Document [Kubernetes Performance](https://docs.google.com/document/d/1-Oefi_zQCNAAysZpSkz4V4H_uQIGf90ZMj79Qevy4zE/edit#).
Once we have enough material the content of the document will be transplanted into this markdown document.
