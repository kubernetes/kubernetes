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
See the [CLI/configuration roadmap](cli-roadmap.md) for details.

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

# Reliability
## Current pain points:
* Writing end-to-end tests should be made easier e.g. not rely so much (or at all) on scripting and as much as possible be written in Go using appropriate frameworks to make it easy to get started with an end-to-end test.
* A developer should be able to take an integration test and turn it into an end to end test (and vice versa) without needing to significantly rewrite the test.
* Some e2e tests currently have false positives (they pass when they should not). 
* It is unclear whether our e2e tests are representative of real workloads. 
* We need to make sure other providers stay healthy as we submit code. Breakages for most providers are found too late.
* Previously discussed: a public dashboard that receives updates from platform maintainers and shows green/red e2e results for each provider per-PR or per-hour or something.
* It is very challenging to bring up large clusters. For example, for GCE, operations that create routes, firewall rules and instances can fail and need to be robustly retried.
* We have no current means to measure the reliability of long running clusters and our current test infrastructure isn’t well suited to this use case.
* We have little or no instrumentation of the various components - memory and CPU usage, time per operation, QPS, etc.
Reliability Goals:
* Automated flow that uses exactly the same source for end-to-end etc. tests from GitHub which can be regularly run (hourly, at commit time etc.) to ensure none of the providers are broken. Comment from Zach: “I think this is "none of the providers we directly support are broken" (GCE, maybe some local, maybe others). The traditional OSS model is that vendors (OpenShift for instance) handle their own downstream testing, unless they're willing to work fully upstream.”
* Dashboard or some other form of storing and querying historical build information.

## Work Items

* Issue [#3130](https://github.com/GoogleCloudPlatform/kubernetes/issues/3130) Rewrite the remaining e2e bash tests in Go. Whilst doing so, reduce/remove the cases where the tests were incorrectly passing.
* Issue [#3131](https://github.com/GoogleCloudPlatform/kubernetes/issues/3131) Refactor the Go e2e tests to use a test framework (ideally just http://golang.org/pkg/testing/ with some extra bits to make sure the cluster is in the right state at the start of the tests). Try to consolidate on a test framework that works the same for integration and e2e tests.
* Issue [#3132](https://github.com/GoogleCloudPlatform/kubernetes/issues/3132) Refactor the e2e tests to allow multiple concurrent runs (assuming it is supported by the cloud provider).
Allow the client to be authenticated to multiple clusters (https://github.com/GoogleCloudPlatform/kubernetes/issues/1755)
* [PR #3046 - done!] Create a GKE cloud provider.
* Issue [#2234](https://github.com/GoogleCloudPlatform/kubernetes/issues/2234) Create an integration test dashboard
* For each supported cloud provider, ensure that we run the e2e tests regularly and fix any breaks
* [done] Setup Jenkins to run on VM/cluster of VMs in GCE. 
* Should have separate projects/flows for testing against different vendors.
* Shared configuration with other GCE projects for vendor specific tests (GKE will need this).
* Issue [#3134](https://github.com/GoogleCloudPlatform/kubernetes/issues/3134) Jenkis should produce build artifacts and push to gcs ~hourly. Ideally we can use this to build and push a ‘continuous’ or ‘latest-dev’ bucket to the official gcs kubernetes-release bucket.
* Issue [#2953]((https://github.com/GoogleCloudPlatform/kubernetes/issues/2953) [zml] Capability bits: I proposed this last week, I still need to write up an issue on it. The idea is that along with the API version (and server version?), the server communicates a bucket of tags that says "I support these capabilities". Then tests like pd.sh can stop being conditionalized on provider and can instead be conditionalized on server capability. Want to get this filed/done before v1beta3, and has testing impact. (Zach edit: The I’s here are me.)
* Stress testing as a Jenkins job using a large-ish number of VMs.
* Issue [#3135](https://github.com/GoogleCloudPlatform/kubernetes/issues/3135) [zml] Upgrade testing: Related to the previous, but you could write an entire doc on upgrade testing alone. I think we're going to need a story here, and it's actually a long one. We need to get a pretty good handle on upgrade/release policy, versions we're going to keep around (OSS-wise, GKE-wise, etc), versions we're going to allow upgrade between, etc. (I volunteer to help pin people down here - I think the release process is getting driven elsewhere but this is a crossbar item between that group and us that's pretty important). (Zach edit: The I’s here are me.)
* Issue [#3136](https://github.com/GoogleCloudPlatform/kubernetes/issues/3136) Create a compatibility test matrix. Verify that an old client works with a new server, different api versions, etc.
* Issue [#3137](https://github.com/GoogleCloudPlatform/kubernetes/issues/3137) Create a soak test. 
* [satnam] Sometimes builds fail after an update and require a build/make-clean.sh. We should ensure that tests, builds etc. get cleaned up properly.
* Issue [#3138](https://github.com/GoogleCloudPlatform/kubernetes/issues/3138) [davidopp] A way to record a real workload and replay it deterministically
* Issue [#3139](https://github.com/GoogleCloudPlatform/kubernetes/issues/3139) [davidopp] A way to generate a synthetic workload and play it
* Issue [#2852](https://github.com/GoogleCloudPlatform/kubernetes/issues/2852) and Issue [#3067](https://github.com/GoogleCloudPlatform/kubernetes/issues/3067) [vishnuk] Protect system services against kernel OOM kills and resource starvation. 
