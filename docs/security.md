# Security in Kubernetes

General design principles and guidelines related to security of containers, APIs, and infrastructure in Kubernetes.


## Objectives

1.  Ensure a clear isolation between container and the underlying host it runs on
2.  Limit the ability of the container to negatively impact the infrastructure or other containers
3.  [Principle of Least Privilege](http://en.wikipedia.org/wiki/Principle_of_least_privilege) - ensure components are only authorized to perform the actions they need, and limit the scope of a compromise by limiting the capabilities of individual components
4.  Reduce the number of systems that have to be hardened and secured by defining clear boundaries between components


## Design Points

### Isolate the data store from the minions and supporting infrastructure

Access to the central data store (etcd) in Kubernetes allows an attacker to run arbitrary containers on hosts, to gain access to any protected information stored in either volumes or in pods (such as access tokens or shared secrets provided as environment variables), to intercept and redirect traffic from running services by inserting middlemen, or to simply delete the entire history of the custer.

As a general principle, access to the central data store should be restricted to the components that need full control over the system and which can apply appropriate authorization and authentication of change requests.  In the future, etcd may offer granular access control, but that granularity will require an administrator to understand the schema of the data to properly apply security.  An administrator must be able to properly secure Kubernetes at a policy level, rather than at an implementation level, and schema changes over time should not risk unintended security leaks.

Both the Kubelet and Kube Proxy need information related to their specific roles - for the Kubelet, the set of pods it should be running, and for the Proxy, the set of services and endpoints to load balance.  The Kubelet also needs to provide information about running pods and historical termination data.  The access pattern for both Kubelet and Proxy to load their configuration is an efficient "wait for changes" request over HTTP.  It should be possible to limit the Kubelet and Proxy to only access the information they need to perform their roles and no more.

The controller manager for Replication Controllers and other future controllers act on behalf of a user via delegation to perform automated maintenance on Kubernetes resources. Their ability to access or modify resource state should be strictly limited to their intended duties and they should be prevented from accessing information not pertinent to their role.  For example, a replication controller needs only to create a copy of a known pod configuration, to determine the running state of an existing pod, or to delete an existing pod that it created - it does not need to know the contents or current state of a pod, nor have access to any data in the pods attached volumes.

The Kubernetes pod scheduler is responsible for reading data from the pod to fit it onto a minion in the cluster.  At a minimum, it needs access to view the ID of a pod (to craft the binding), its current state, any resource information necessary to identify placement, and other data relevant to concerns like anti-affinity, zone or region preference, or custom logic.  It does not need the ability to modify pods or see other resources, only to create bindings.  It should not need the ability to delete bindings unless the scheduler takes control of relocating components on failed hosts (which could be implemented by a separate component that can delete bindings but not create them).  The scheduler may need read access to user or project-container information to determine preferential location (underspecified at this time).