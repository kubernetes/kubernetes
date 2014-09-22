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
   - K8s Cluster Admins: People who control the machines, networks, or binaries that comprise a K8s cluster.
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
   - Personal information like private messages, or images uploaded by External Users
   - web server logs

K8s User assets:
  - External User assets of each K8s User
  - things private to the K8s app, like:
    - credentials for accessing other services (docker private repos, storage services, facebook, etc)
    - SSL certificates for web servers
    - proprietary data and code

K8s Cluster assets:
  - Assets of each K8s User
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
   - Minions may be on-premises VMs or physical machines; Cloud VMs; or a mix.

Hosted cluster:
  - Offering K8s API as a service, or offering a Paas or Saas built on K8s
  - May already offer web services, and need to integrate with existing customer account concept, and existing authentication, accounting, auditing, and security policy infrastructure.
  - May want to leverage K8s User accounts and accounting to manage their User accounts (not a priority to support this use case.)
  - Precise and accurate accounting of resources needed.  Resource controls needed for hard limits (Users given limited slice of data) and soft limits (Users can grow up to some limit and then be expanded).

K8s ecosystem services:
 - There may be companies that want to offer their existing services (Build, CI, A/B-test, release automation, etc) for use with K8s.  There should be some story for this case.

Pods configs should be largely portable between Org-run and hosted configurations.


# Design
Related discussion:
- https://github.com/GoogleCloudPlatform/kubernetes/issues/442
- https://github.com/GoogleCloudPlatform/kubernetes/issues/443

## Identity
K8s itself does not have REST resources for users, groups or authorization policies.  The Authorization [Plugins](proposals/auth_plugins.md) does make reference to these concepts and [API Plugins](#1355) may provide REST resources for users and groups for some implementation of auth.

###Unix accounts 
The fact that a pod is started by a particular authenticated user does not mean that the processes in that pod's containers run as a Unix user with a corresponding name or identity.  

Initially:
- The unix accounts available in a container, and used by the processes running in a container are those that are provided by the base operating system and the Docker manifest. 

Improvements:
- Kubelet allocates disjoint blocks of root-namespace uids for each container.  This may provide some defense-in-depth against container escapes. (https://github.com/docker/docker/pull/4572)
- requires docker to integrate user namespace support, and deciding what getpwnam() does for these uids.
- any features that help users avoid use of privileged containers (https://github.com/GoogleCloudPlatform/kubernetes/issues/391)

###Namespaces
K8s will have a have a `namespace` API object.  It is similar to a Google Compute Engine `project`.  It provides a namespace for objects created by a group of people co-operating together, preventing name collisions with non-cooperating groups.  It also serves as a reference point for authorization policies.

Namespaces are described in [namespace.md](namespaces.md).

## Authentication

Goals for K8s authentication:
- Include a built-in authentication system with no configuration required to use in single-user mode, and little configuration required to add several user accounts, and no https proxy required.
- Allow for authentication to be handled by a system external to Kubernetes, to allow integration with existing to enterprise authorization systems.  The kubernetes namespace itself should avoid taking contributions of multiple authorization schemes.  Instead, a trusted proxy in front of the apiserver can be used to authenticate users.
  - For organizations whose security requirements only allow FIPS compliant implementations (e.g. apache) for authentication.
  - So the proxy can terminate SSL, and isolate the CA-signed certificate from less trusted, higher-touch APIserver.
  - For organizations that already have existing SaaS web services (e.g. storage, VMs) and want a common authentication portal.
- Avoid mixing authentication and authorization, so that authorization policies be centrally managed, and to allow changes in authentication methods without affecting authorization code.

Initially:
- Tokens used to authenticate a user.
<<<<<<< HEAD:docs/design/access.md
- Long lived tokens identify a particular `userAccount`.
- Administrator utility generates tokens at cluster setup.
=======
- Long lived tokens identify a particular `userAccount`.  
>>>>>>> Added proposal for how Auth plugins work.  Remove conflicting words from access.md.:docs/access.md
- OAuth2.0 Bearer tokens protocol, http://tools.ietf.org/html/rfc6750
- No scopes for tokens.  Authorization happens in the API server
- Tokens dynamically generated by apiserver to identify pods which are making API calls.
- Tokens checked in a module of the APIserver.
- Authentication in apiserver can be disabled by flag, to allow testing without authorization enabled, and to allow use of an authenticating proxy.  In this mode, a query parameter or header added by the proxy will identify the caller.

To be considered for subsequent versions:
- Fuller use of OAuth (http://tools.ietf.org/html/rfc6749)
- Scoped tokens.
- Tokens that are bound to the channel between the client and the api server
     - http://www.ietf.org/proceedings/90/slides/slides-90-uta-0.pdf
     - http://www.browserauth.net


See [Auth plugins](./docs/proposals/auth_plugins.md).

## Authorization

K8s authorization should:
- Allow for a range of maturity levels, from single-user for those test driving the system, to integration with existing to enterprise authorization systems.
- Allow for centralized management of users and policies.  In some organizations, this will mean that the definition of users and access policies needs to reside on a system other than k8s and encompass other web services (such as a storage service).
- Allow processes running in K8s Pods to take on identity, and to allow narrow scoping of permissions for those identities in order to limit damage from software faults.
- Have Authorization Policies exposed as API objects so that a single config file can create or delete Pods, Controllers, Services, and the identities and policies for those Pods and Controllers.
- Be separate as much as practical from Authentication, to allow Authentication methods to change over time and space, without impacting Authorization policies.

See [Auth plugins](./docs/proposals/auth_plugins.md).

## Accounting

The API should have a `quota` concept (see https://github.com/GoogleCloudPlatform/kubernetes/issues/442).  A quota object relates a namespace (and optionally a label selector) to a maximum quantity of resources that may be used (see [resources.md](/docs/resources.md)).

Initially:
- a `quota` object is immutable.
- for hosted K8s systems that do billing, Project is recommended level for billing accounts.
- Every object that consumes resources should have a `namespace` so that Resource usage stats are roll-up-able to `namespace`.
- K8s Cluster Admin sets quota objects by writing a config file.

Improvements:
- allow one namespace to charge the quota for one or more other namespaces.  This would be controlled by a policy which allows changing a billing_namespace= label on an object.
- allow quota to be set by namespace owners for (namespace x label) combinations (e.g. let "webserver" namespace use 100 cores, but to prevent accidents, don't allow "webserver" namespace and "instance=test" use more than 10 cores.
- tools to help write consistent quota config files based on number of minions, historical namespace usages, QoS needs, etc.
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
