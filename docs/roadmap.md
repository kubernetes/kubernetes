# Kubernetes Roadmap

Updated August 8, 2014

This document is intended to capture the set of features, docs, and patterns that we feel are required to call Kubernetes “feature complete” for a 1.0 release candidate.  This list does not emphasize the bug fixes and stabilization that will be required to take it all the way to production ready.  This is a living document, and is certainly open for discussion.

## APIs
1. Versioned APIs:  Manage APIs for master components and kubelets with explicit versions, version-specific conversion routines, and component-to-component version checking.
2. Deprecation policy: Declare the project’s intentions with regards to expiring and removing features and interfaces.
3. Compatibility policy: Declare the project’s intentions with regards to saved state and live upgrades of components.
4. Component-centric APIs:  Clarify which types belong in each component’s API and which ones are truly common.
5. Idempotency: Whenever possible APIs must be idempotent.
6. Container restart policy: Policy for each pod or container stating whether and when it should be restarted upon termination.
7. Life cycle events/hooks and notifications: Notify containers about what is happening to them.
8. Re-think the network parts of the API: Find resolution on the the multiple issues around networking.
  1. Using the host network
  2. Representation of Ports in the Manifest structure
  3. Utility of HostPorts in ip-per-pod
  4. Scenarios where IP-per-pod is hard or impossible
  5. Port collisions between services
9. Provide a model for durable local volumes including scheduler constraints.
10. Auth[nz] and ACLs: Have a plan for how identity, authentication, and authorization will fit in to the API, as well as ACLs for objects, and basic resource quotas.
  1. Projects / subdivision: Have a plan for how security isolation between users could apply in terms of grouping resources (calling out explicitly) and whether there is a common model that could apply to Kubernetes


## Factoring and pluggability
1. Pluggable scheduling: Cleanly separate the scheduler from the apiserver.
2. Pluggable naming and discovery: Call-outs or hooks to enable external naming systems.
3. Pluggable volumes: Allow new kinds of data sources as volumes.
4. Replication controller: Make replication controller a standalone entity in the master stack.
5. Pod templates: Proposal to make pod templates a first-class API object, rather than an artifact of replica controller
6. Auto-scaling controller: Make a sizing controller, canary controller. Probably want to have a source of QPS and error rate information for an application first.
7. Pluggable authentication, with identity and authorization being dependent on auth[nz] above

## Cluster features
1. Minion death: Cleanly handle the loss of a minion.
2. Configure DNS: Provide DNS service for k8s running pods, containers and services. Auto-populate it with the things we know.
3. Resource requirements and scheduling: Use knowledge of resources available and resources required to do better scheduling.
4. IP-per-service: Proposal to make proxies less necessary.
5. Pod spreading: Scheduler spreads pods for higher availability.
6. Basic deployment tools.
7. Standard mechanisms for deploying k8s on k8s with a clear strategy for reusing the infrastructure for self-host.

## Node features
1. Container termination reasons: Capture and report exit codes and other termination reasons.
2. Container status snippets: Capture and report app-specific status snippets.
3. Garbage collect old container images: Clean up old docker images that consume local disk. Maybe a TTL on images.
4. Container logs: Expose stdout/stderr from containers without users having to SSH into minions.  Needs a rotation policy to avoid disks getting filled.
5. Container performance information: Capture and report performance data for each container.
6. Plan for working with upstream Docker on the Docker-daemon-kills-all-children-on-exit problem.

## Global features
1. True IP-per-pod: Get rid of last remnants of shared port spaces.
2. Input validation: Stop bad input as early as possible.
3. Error propagation: Report problems reliably and consistently.

## Patterns and specifications
1. Naming/discovery: Make it possible for common patterns to operate:
  1. Master-elected services
  2. DB replicas
  3. Sharded services
  4. Worker pools
2. Interconnection of services: expand / decompose the service pattern to take into account:
  1. Network boundaries - private / public
  2. Allow external or shared load balancers across a deployment to be registered (name based balancers)
  3. Registering DNS name balancing
3. Networking: Well documented recipes for settings where the networking is not the same as GCE.
4. Health-checking: Specification for how it works and best practices.
5. Logging: Well documented recipes for setting up log collection.
6. Rolling updates: Demo and best practices for live application upgrades.
  1. Have a plan for how higher level deployment / update concepts should / should not fit into Kubernetes
7. Minion requirements: Document the requirements and integrations between kubelet and minion machine environments.

