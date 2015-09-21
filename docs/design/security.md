<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/design/security.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Security in Kubernetes

Kubernetes should define a reasonable set of security best practices that allows processes to be isolated from each other, from the cluster infrastructure, and which preserves important boundaries between those who manage the cluster, and those who use the cluster.

While Kubernetes today is not primarily a multi-tenant system, the long term evolution of Kubernetes will increasingly rely on proper boundaries between users and administrators. The code running on the cluster must be appropriately isolated and secured to prevent malicious parties from affecting the entire cluster.


## High Level Goals

1.  Ensure a clear isolation between the container and the underlying host it runs on
2.  Limit the ability of the container to negatively impact the infrastructure or other containers
3.  [Principle of Least Privilege](http://en.wikipedia.org/wiki/Principle_of_least_privilege) - ensure components are only authorized to perform the actions they need, and limit the scope of a compromise by limiting the capabilities of individual components
4.  Reduce the number of systems that have to be hardened and secured by defining clear boundaries between components
5.  Allow users of the system to be cleanly separated from administrators
6.  Allow administrative functions to be delegated to users where necessary
7.  Allow applications to be run on the cluster that have "secret" data (keys, certs, passwords) which is properly abstracted from "public" data.


## Use cases

### Roles

We define "user" as a unique identity accessing the Kubernetes API server, which may be a human or an automated process.  Human users fall into the following categories:

1. k8s admin - administers a Kubernetes cluster and has access to the underlying components of the system
2. k8s project administrator - administrates the security of a small subset of the cluster
3. k8s developer - launches pods on a Kubernetes cluster and consumes cluster resources

Automated process users fall into the following categories:

1. k8s container user - a user that processes running inside a container (on the cluster) can use to access other cluster resources independent of the human users attached to a project
2. k8s infrastructure user - the user that Kubernetes infrastructure components use to perform cluster functions with clearly defined roles


### Description of roles

* Developers:
  * write pod specs.
  * making some of their own images, and using some "community" docker images
  * know which pods need to talk to which other pods
  * decide which pods should share files with other pods, and which should not.
  * reason about application level security, such as containing the effects of a local-file-read exploit in a webserver pod.
  * do not often reason about operating system or organizational security.
  * are not necessarily comfortable reasoning about the security properties of a system at the level of detail of Linux Capabilities, SELinux, AppArmor, etc.

* Project Admins:
  * allocate identity and roles within a namespace
  * reason about organizational security within a namespace
    * don't give a developer permissions that are not needed for role.
    * protect files on shared storage from unnecessary cross-team access
  * are less focused about application security

* Administrators:
  * are less focused on application security. Focused on operating system security.
  * protect the node from bad actors in containers, and properly-configured innocent containers from bad actors in other containers.
  * comfortable reasoning about the security properties of a system at the level of detail of Linux Capabilities, SELinux, AppArmor, etc.
  * decides who can use which Linux Capabilities, run privileged containers, use hostPath, etc.
    * e.g. a team that manages Ceph or a mysql server might be trusted to have raw access to storage devices in some organizations, but teams that develop the applications at higher layers would not.


## Proposed Design

A pod runs in a *security context* under a *service account* that is defined by an administrator or project administrator, and the *secrets* a pod has access to is limited by that *service account*.


1. The API should authenticate and authorize user actions [authn and authz](access.md)
2. All infrastructure components (kubelets, kube-proxies, controllers, scheduler) should have an infrastructure user that they can authenticate with and be authorized to perform only the functions they require against the API.
3. Most infrastructure components should use the API as a way of exchanging data and changing the system, and only the API should have access to the underlying data store (etcd)
4. When containers run on the cluster and need to talk to other containers or the API server, they should be identified and authorized clearly as an autonomous process via a [service account](service_accounts.md)
   1.  If the user who started a long-lived process is removed from access to the cluster, the process should be able to continue without interruption
   2.  If the user who started processes are removed from the cluster, administrators may wish to terminate their processes in bulk
   3.  When containers run with a service account, the user that created / triggered the service account behavior must be associated with the container's action
5. When container processes run on the cluster, they should run in a [security context](security_context.md) that isolates those processes via Linux user security, user namespaces, and permissions.
   1.  Administrators should be able to configure the cluster to automatically confine all container processes as a non-root, randomly assigned UID
   2.  Administrators should be able to ensure that container processes within the same namespace are all assigned the same unix user UID
   3.  Administrators should be able to limit which developers and project administrators have access to higher privilege actions
   4.  Project administrators should be able to run pods within a namespace under different security contexts, and developers must be able to specify which of the available security contexts they may use
   5.  Developers should be able to run their own images or images from the community and expect those images to run correctly
   6.  Developers may need to ensure their images work within higher security requirements specified by administrators
   7.  When available, Linux kernel user namespaces can be used to ensure 5.2 and 5.4 are met.
   8.  When application developers want to share filesystem data via distributed filesystems, the Unix user ids on those filesystems must be consistent across different container processes
6. Developers should be able to define [secrets](secrets.md) that are automatically added to the containers when pods are run
   1.  Secrets are files injected into the container whose values should not be displayed within a pod. Examples:
       1. An SSH private key for git cloning remote data
       2. A client certificate for accessing a remote system
       3. A private key and certificate for a web server
       4. A .kubeconfig file with embedded cert / token data for accessing the Kubernetes master
       5. A .dockercfg file for pulling images from a protected registry
   2.  Developers should be able to define the pod spec so that a secret lands in a specific location
   3.  Project administrators should be able to limit developers within a namespace from viewing or modifying secrets (anyone who can launch an arbitrary pod can view secrets)
   4.  Secrets are generally not copied from one namespace to another when a developer's application definitions are copied


### Related design discussion

* [Authorization and authentication](access.md)
* [Secret distribution via files](http://pr.k8s.io/2030)
* [Docker secrets](https://github.com/docker/docker/pull/6697)
* [Docker vault](https://github.com/docker/docker/issues/10310)
* [Service Accounts:](service_accounts.md)
* [Secret volumes](http://pr.k8s.io/4126)

## Specific Design Points

### TODO: authorization, authentication

### Isolate the data store from the nodes and supporting infrastructure

Access to the central data store (etcd) in Kubernetes allows an attacker to run arbitrary containers on hosts, to gain access to any protected information stored in either volumes or in pods (such as access tokens or shared secrets provided as environment variables), to intercept and redirect traffic from running services by inserting middlemen, or to simply delete the entire history of the custer.

As a general principle, access to the central data store should be restricted to the components that need full control over the system and which can apply appropriate authorization and authentication of change requests.  In the future, etcd may offer granular access control, but that granularity will require an administrator to understand the schema of the data to properly apply security.  An administrator must be able to properly secure Kubernetes at a policy level, rather than at an implementation level, and schema changes over time should not risk unintended security leaks.

Both the Kubelet and Kube Proxy need information related to their specific roles - for the Kubelet, the set of pods it should be running, and for the Proxy, the set of services and endpoints to load balance.  The Kubelet also needs to provide information about running pods and historical termination data.  The access pattern for both Kubelet and Proxy to load their configuration is an efficient "wait for changes" request over HTTP.  It should be possible to limit the Kubelet and Proxy to only access the information they need to perform their roles and no more.

The controller manager for Replication Controllers and other future controllers act on behalf of a user via delegation to perform automated maintenance on Kubernetes resources. Their ability to access or modify resource state should be strictly limited to their intended duties and they should be prevented from accessing information not pertinent to their role.  For example, a replication controller needs only to create a copy of a known pod configuration, to determine the running state of an existing pod, or to delete an existing pod that it created - it does not need to know the contents or current state of a pod, nor have access to any data in the pods attached volumes.

The Kubernetes pod scheduler is responsible for reading data from the pod to fit it onto a node in the cluster.  At a minimum, it needs access to view the ID of a pod (to craft the binding), its current state, any resource information necessary to identify placement, and other data relevant to concerns like anti-affinity, zone or region preference, or custom logic.  It does not need the ability to modify pods or see other resources, only to create bindings.  It should not need the ability to delete bindings unless the scheduler takes control of relocating components on failed hosts (which could be implemented by a separate component that can delete bindings but not create them).  The scheduler may need read access to user or project-container information to determine preferential location (underspecified at this time).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/security.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
