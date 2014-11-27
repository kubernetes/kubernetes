# Kubernetes Roadmap

Updated August 28, 2014

This document is intended to capture the set of features, docs, and patterns that we feel are required to call Kubernetes “feature complete” for a 1.0 release candidate.  This list does not emphasize the bug fixes and stabilization that will be required to take it all the way to production ready.  This is a living document, and is certainly open for discussion.

## APIs
1. ~~Versioned APIs:  Manage APIs for master components and kubelets with explicit versions, version-specific conversion routines, and component-to-component version checking.~~ **Done**
2. Component-centric APIs:  Clarify which types belong in each component’s API and which ones are truly common.
  1. Clarify the role of etcd in the cluster.
3. Idempotency: Whenever possible APIs must be idempotent.
4. Container restart policy: Policy for each pod or container stating whether and when it should be restarted upon termination.
5. Life cycle events/hooks and notifications: Notify containers about what is happening to them.
6. Re-think the network parts of the API: Find resolution on the the multiple issues around networking.
  1. ~~Utility of HostPorts in ip-per-pod~~ **Done**
  2. Services/Links/Portals/Ambassadors
7. Durable volumes: Provide a model for data that survives some kinds of outages.
8. Auth[nz] and ACLs: Have a plan for how the API and system will express:
  1. Identity & authentication
  2. Authorization & access control
  3. Cluster subdivision, accounting, & isolation

## Factoring and pluggability
1. ~~Pluggable scheduling: Cleanly separate the scheduler from the apiserver.~~ **Done**
2. Pluggable naming and discovery: Call-outs or hooks to enable external naming systems.
3. Pluggable volumes: Allow new kinds of data sources as volumes.
4. Replication controller: Make replication controller a standalone entity in the master stack.
5. Pod templates: Proposal to make pod templates a first-class API object, rather than an artifact of replica controller

## Cluster features
1. ~~Minion death: Cleanly handle the loss of a minion.~~ **Done**
2. Configure DNS: Provide DNS service for k8s running pods, containers and services. Auto-populate it with the things we know.
3. Resource requirements and scheduling: Use knowledge of resources available and resources required to do better scheduling.
4. ~~True IP-per-pod: Get rid of last remnants of shared port spaces for pods.~~ **Done**
5. IP-per-service: Proposal to make services cleaner.
6. Basic deployment tools: This includes tools for higher-level deployments configs.
7. Standard mechanisms for deploying k8s on k8s with a clear strategy for reusing the infrastructure for self-host.

## Node features
1. Container termination reasons: Capture and report exit codes and other termination reasons.
2. Garbage collect old container images: Clean up old docker images that consume local disk. Maybe a TTL on images.
3. Container logs: Expose stdout/stderr from containers without users having to SSH into minions.  Needs a rotation policy to avoid disks getting filled.
4. Container performance information: Capture and report performance data for each container.
5. Host log management: Make sure we don't kill nodes with full disks.

## Global features
2. Input validation: Stop bad input as early as possible.
3. Error propagation: Report problems reliably and consistently.
4. Consistent patterns of usage of IDs and names throughout the system.
5. Binary release: Repeatable process to produce binaries for release.

## Patterns, policies, and specifications
1. Deprecation policy: Declare the project’s intentions with regards to expiring and removing features and interfaces.
2. Compatibility policy: Declare the project’s intentions with regards to saved state and live upgrades of components.
3. Naming/discovery: Demonstrate techniques for common patterns:
  1. Master-elected services
  2. DB replicas
  3. Sharded services
  4. Worker pools
4. Health-checking: Specification for how it works and best practices.
5. Logging: Demonstrate setting up log collection.
6. ~~Monitoring: Demonstrate setting up cluster monitoring.~~ **Done**
7. Rolling updates: Demo and best practices for live application upgrades.
  1. Have a plan for how higher level deployment / update concepts should / should not fit into Kubernetes
8. Minion requirements: Document the requirements and integrations between kubelet and minion machine environments.
