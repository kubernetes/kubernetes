<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# K8s Identity and Access Management Sketch

This document suggests a direction for identity and access management in the Kubernetes system.


## Background

High level goals are:
   - Have a plan for how identity, authentication, and authorization will fit in to the API.
   - Have a plan for partitioning resources within a cluster between independent organizational units.
   - Ease integration with existing enterprise and hosted scenarios.

### Actors

Each of these can act as normal users or attackers.
   - External Users: People who are accessing applications running on K8s (e.g. a web site served by webserver running in a container on K8s), but who do not have K8s API access.
   - K8s Users : People who access the K8s API (e.g. create K8s API objects like Pods)
   - K8s Project Admins: People who manage access for some K8s Users
   - K8s Cluster Admins: People who control the machines, networks, or binaries that make up a K8s cluster.
   - K8s Admin means K8s Cluster Admins and K8s Project Admins taken together.

### Threats

Both intentional attacks and accidental use of privilege are concerns.

For both cases it may be useful to think about these categories differently:
  - Application Path - attack by sending network messages from the internet to the IP/port of any application running on K8s.  May exploit weakness in application or misconfiguration of K8s.
  - K8s API Path - attack by sending network messages to any K8s API endpoint.
  - Insider Path - attack on K8s system components.  Attacker may have privileged access to networks, machines or K8s software and data.  Software errors in K8s system components and administrator error are some types of threat in this category.

This document is primarily concerned with K8s API paths, and secondarily with Internal paths.   The Application path also needs to be secure, but is not the focus of this document.

### Assets to protect

External User assets:
   - Personal information like private messages, or images uploaded by External Users.
   - web server logs.

K8s User assets:
  - External User assets of each K8s User.
  - things private to the K8s app, like:
    - credentials for accessing other services (docker private repos, storage services, facebook, etc)
    - SSL certificates for web servers
    - proprietary data and code

K8s Cluster assets:
  - Assets of each K8s User.
  - Machine Certificates or secrets.
  - The value of K8s cluster computing resources (cpu, memory, etc).

This document is primarily about protecting K8s User assets and K8s cluster assets from other K8s Users and K8s Project and Cluster Admins.

### Usage environments

Cluster in Small organization:
   - K8s Admins may be the same people as K8s Users.
   - few K8s Admins.
    - prefer ease of use to fine-grained access control/precise accounting, etc.
 - Product requirement that it be easy for potential K8s Cluster Admin to try out setting up a simple cluster.

Cluster in Large organization:
   - K8s Admins typically distinct people from K8s Users.  May need to divide K8s Cluster Admin access by roles.
   - K8s Users need to be protected from each other.
   - Auditing of K8s User and K8s Admin actions important.
   - flexible accurate usage accounting and resource controls important.
   - Lots of automated access to APIs.
   - Need to integrate with existing enterprise directory, authentication, accounting, auditing, and security policy infrastructure.

Org-run cluster:
   - organization that runs K8s master components is same as the org that runs apps on K8s.
   - Nodes may be on-premises VMs or physical machines; Cloud VMs; or a mix.

Hosted cluster:
  - Offering K8s API as a service, or offering a Paas or Saas built on K8s.
  - May already offer web services, and need to integrate with existing customer account concept, and existing authentication, accounting, auditing, and security policy infrastructure.
  - May want to leverage K8s User accounts and accounting to manage their User accounts (not a priority to support this use case.)
  - Precise and accurate accounting of resources needed.  Resource controls needed for hard limits (Users given limited slice of data) and soft limits (Users can grow up to some limit and then be expanded).

K8s ecosystem services:
 - There may be companies that want to offer their existing services (Build, CI, A/B-test, release automation, etc) for use with K8s.  There should be some story for this case.

Pods configs should be largely portable between Org-run and hosted configurations.


# Design

Related discussion:
- http://issue.k8s.io/442
- http://issue.k8s.io/443

This doc describes two security profiles:
  - Simple profile:  like single-user mode.  Make it easy to evaluate K8s without lots of configuring accounts and policies.  Protects from unauthorized users, but does not partition authorized users.
  - Enterprise profile: Provide mechanisms needed for large numbers of users.  Defense in depth.  Should integrate with existing enterprise security infrastructure.

K8s distribution should include templates of config, and documentation, for simple and enterprise profiles.  System should be flexible enough for knowledgeable users to create intermediate profiles, but K8s developers should only reason about those two Profiles, not a matrix.

Features in this doc are divided into "Initial Feature", and "Improvements".   Initial features would be candidates for version 1.00.

## Identity

### userAccount

K8s will have a `userAccount` API object.
- `userAccount` has a UID which is immutable.  This is used to associate users with objects and to record actions in audit logs.
- `userAccount` has a name which is a string and human readable and unique among userAccounts.  It is used to refer to users in Policies, to ensure that the Policies are human readable.  It can be changed only when there are no Policy objects or other objects which refer to that name.  An email address is a suggested format for this field.
- `userAccount` is not related to the unix username of processes in Pods created by that userAccount.
- `userAccount` API objects can have labels.

The system may associate one or more Authentication Methods with a
`userAccount` (but they are not formally part of the userAccount object.)
In a simple deployment, the authentication method for a
user might be an authentication token which is verified by a K8s server.  In a
more complex deployment, the authentication might be delegated to
another system which is trusted by the K8s API to authenticate users, but where
the authentication details are unknown to K8s.

Initial Features:
- there is no superuser `userAccount`
- `userAccount` objects are statically populated in the K8s API store by reading a config file.  Only a K8s Cluster Admin can do this.
- `userAccount` can have a default `namespace`.  If API call does not specify a `namespace`, the default `namespace` for that caller is assumed.
- `userAccount` is global.  A single human with access to multiple namespaces is recommended to only have one userAccount.

Improvements:
- Make `userAccount` part of a separate API group from core K8s objects like `pod`.  Facilitates plugging in alternate Access Management.

Simple Profile:
   - single `userAccount`, used by all K8s Users and Project Admins.  One access token shared by all.

Enterprise Profile:
   - every human user has own `userAccount`.
   - `userAccount`s have labels that indicate both membership in groups, and ability to act in certain roles.
   - each service using the API has own `userAccount` too. (e.g. `scheduler`, `repcontroller`)
   - automated jobs to denormalize the ldap group info into the local system list of users into the K8s userAccount file.

### Unix accounts

A `userAccount` is not a Unix user account.  The fact that a pod is started by a `userAccount` does not mean that the processes in that pod's containers run as a Unix user with a corresponding name or identity.

Initially:
- The unix accounts available in a container, and used by the processes running in a container are those that are provided by the combination of the base operating system and the Docker manifest.
- Kubernetes doesn't enforce any relation between `userAccount` and unix accounts.

Improvements:
- Kubelet allocates disjoint blocks of root-namespace uids for each container.  This may provide some defense-in-depth against container escapes. (https://github.com/docker/docker/pull/4572)
- requires docker to integrate user namespace support, and deciding what getpwnam() does for these uids.
- any features that help users avoid use of privileged containers (http://issue.k8s.io/391)

### Namespaces

K8s will have a have a `namespace` API object.  It is similar to a Google Compute Engine `project`.  It provides a namespace for objects created by a group of people co-operating together, preventing name collisions with non-cooperating groups.  It also serves as a reference point for authorization policies.

Namespaces are described in [namespaces.md](namespaces.md).

In the Enterprise Profile:
   - a `userAccount` may have permission to access several `namespace`s.

In the Simple Profile:
   - There is a single `namespace` used by the single user.

Namespaces versus userAccount vs Labels:
- `userAccount`s are intended for audit logging (both name and UID should be logged), and to define who has access to `namespace`s.
- `labels` (see [docs/user-guide/labels.md](../../docs/user-guide/labels.md)) should be used to distinguish pods, users, and other objects that cooperate towards a common goal but are different in some way, such as version, or responsibilities.
- `namespace`s prevent name collisions between uncoordinated groups of people, and provide a place to attach common policies for co-operating groups of people.


## Authentication

Goals for K8s authentication:
- Include a built-in authentication system with no configuration required to use in single-user mode, and little configuration required to add several user accounts, and no https proxy required.
- Allow for authentication to be handled by a system external to Kubernetes, to allow integration with existing to enterprise authorization systems.  The Kubernetes namespace itself should avoid taking contributions of multiple authorization schemes.  Instead, a trusted proxy in front of the apiserver can be used to authenticate users.
  - For organizations whose security requirements only allow FIPS compliant implementations (e.g. apache) for authentication.
  - So the proxy can terminate SSL, and isolate the CA-signed certificate from less trusted, higher-touch APIserver.
  - For organizations that already have existing SaaS web services (e.g. storage, VMs) and want a common authentication portal.
- Avoid mixing authentication and authorization, so that authorization policies be centrally managed, and to allow changes in authentication methods without affecting authorization code.

Initially:
- Tokens used to authenticate a user.
- Long lived tokens identify a particular `userAccount`.
- Administrator utility generates tokens at cluster setup.
- OAuth2.0 Bearer tokens protocol, http://tools.ietf.org/html/rfc6750
- No scopes for tokens.  Authorization happens in the API server
- Tokens dynamically generated by apiserver to identify pods which are making API calls.
- Tokens checked in a module of the APIserver.
- Authentication in apiserver can be disabled by flag, to allow testing without authorization enabled, and to allow use of an authenticating proxy.  In this mode, a query parameter or header added by the proxy will identify the caller.

Improvements:
- Refresh of tokens.
- SSH keys to access inside containers.

To be considered for subsequent versions:
- Fuller use of OAuth (http://tools.ietf.org/html/rfc6749)
- Scoped tokens.
- Tokens that are bound to the channel between the client and the api server
     - http://www.ietf.org/proceedings/90/slides/slides-90-uta-0.pdf
     - http://www.browserauth.net


## Authorization

K8s authorization should:
- Allow for a range of maturity levels, from single-user for those test driving the system, to integration with existing to enterprise authorization systems.
- Allow for centralized management of users and policies.  In some organizations, this will mean that the definition of users and access policies needs to reside on a system other than k8s and encompass other web services (such as a storage service).
- Allow processes running in K8s Pods to take on identity, and to allow narrow scoping of permissions for those identities in order to limit damage from software faults.
- Have Authorization Policies exposed as API objects so that a single config file can create or delete Pods, Replication Controllers, Services, and the identities and policies for those Pods and Replication Controllers.
- Be separate as much as practical from Authentication, to allow Authentication methods to change over time and space, without impacting Authorization policies.

K8s will implement a relatively simple
[Attribute-Based Access Control](http://en.wikipedia.org/wiki/Attribute_Based_Access_Control) model.
The model will be described in more detail in a forthcoming document.  The model will
- Be less complex than XACML
- Be easily recognizable to those familiar with Amazon IAM Policies.
- Have a subset/aliases/defaults which allow it to be used in a way comfortable to those users more familiar with Role-Based Access Control.

Authorization policy is set by creating a set of Policy objects.

The API Server will be the Enforcement Point for Policy. For each API call that it receives, it will construct the Attributes needed to evaluate the policy (what user is making the call, what resource they are accessing, what they are trying to do that resource, etc) and pass those attributes to a Decision Point.  The Decision Point code evaluates the Attributes against all the Policies and allows or denies the API call.  The system will be modular enough that the Decision Point code can either be linked into the APIserver binary, or be another service that the apiserver calls for each Decision (with appropriate time-limited caching as needed for performance).

Policy objects may be applicable only to a single namespace or to all namespaces; K8s Project Admins would be able to create those as needed.  Other Policy objects may be applicable to all namespaces; a K8s Cluster Admin might create those in order to authorize a new type of controller to be used by all namespaces, or to make a K8s User into a K8s Project Admin.)


## Accounting

The API should have a `quota` concept (see http://issue.k8s.io/442).  A quota object relates a namespace (and optionally a label selector) to a maximum quantity of resources that may be used (see [resources design doc](resources.md)).

Initially:
- a `quota` object is immutable.
- for hosted K8s systems that do billing, Project is recommended level for billing accounts.
- Every object that consumes resources should have a `namespace` so that Resource usage stats are roll-up-able to `namespace`.
- K8s Cluster Admin sets quota objects by writing a config file.

Improvements:
- allow one namespace to charge the quota for one or more other namespaces.  This would be controlled by a policy which allows changing a billing_namespace= label on an object.
- allow quota to be set by namespace owners for (namespace x label) combinations (e.g. let "webserver" namespace use 100 cores, but to prevent accidents, don't allow "webserver" namespace and "instance=test" use more than 10 cores.
- tools to help write consistent quota config files based on number of nodes, historical namespace usages, QoS needs, etc.
- way for K8s Cluster Admin to incrementally adjust Quota objects.

Simple profile:
   - a single `namespace` with infinite resource limits.

Enterprise profile:
   - multiple namespaces each with their own limits.

Issues:
- need for locking or "eventual consistency" when multiple apiserver goroutines are accessing the object store and handling pod creations.


## Audit Logging

API actions can be logged.

Initial implementation:
- All API calls logged to nginx logs.

Improvements:
- API server does logging instead.
- Policies to drop logging for high rate trusted API calls, or by users performing audit or other sensitive functions.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/access.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
