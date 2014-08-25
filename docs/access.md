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

This doc describes two security profiles:
  - Simple profile:  like single-user mode.  Make it easy to evaluate K8s without lots of configuring accounts and policies.  Protects from unauthorized users, but does not partition authorized users.
  - Enterprise profile: Provide mechanisms needed for large numbers of users.  Defence in depth.  Should integrate with existing enterprise security infrastructure.
 
K8s distribution should include templates of config, and documentation, for simple and enterprise profiles.  System should be flexible enough for knowledgeable users to create intermediate profiles, but K8s developers should only reason about those two Profiles, not a matrix.

Features in this doc are divided into "Initial Feature", and "Improvements".   Initial features would be candidates for version 1.00.

## Identity
###userAccount
K8s will have a `userAccount` API object.
- `userAccount` has a UID which is immutable.  This is used to associate users with objects and to record actions in audit logs.
- `userAccount` has a name which is a string and human readable and unique among userAccounts.  It is used to refer to users in Policies, to ensure that the Policies are human readable.  It can be changed only when there are no Policy objects or other objects which refer to that name.  An email address is a suggested format for this field.
- `userAccount` is not related to the unix username of processes in Pods created by that userAccount.
- `userAccount` API objects can have labels

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
- `userAccount` can have a default `project`.  If API call does not specify a `project`, the default `project` for that caller is assumed.
- `userAccount` is global.  A single human with access to multiple projects is recommended to only have one userAccount.

Improvements:
- Make `userAccount` part of a separate API group from core K8s objects like `pod`.  Facilitates plugging in alternate Access Management.

Simple Profile:
   - single `userAccount`, used by all K8s Users and Project Admins.  One access token shared by all.

Enterprise Profile:
   - every human user has own `userAccount`. 
   - `userAccount`s have labels that indicate both membership in groups, and ability to act in certain roles.
   - each service using the API has own `userAccount` too. (e.g. `scheduler`, `repcontroller`)
   - automated jobs to denormalize the ldap group info into the local system list of users into the K8s userAccount file.

###Unix accounts 
A `userAccount` is not a Unix user account.  The fact that a pod is started by a `userAccount` does not mean that the processes in that pod's containers run as a Unix user with a corresponding name or identity.  

Initially:
- The unix accounts available in a container, and used by the processes running in a container are those that are provided by the combination of the base operating system and the Docker manifest. 
- Kubernetes doesn't enforce any relation between `userAccount` and unix accounts.

Improvements:
- Kubelet allocates disjoint blocks of root-namespace uids for each container.  This may provide some defense-in-depth against container escapes. (https://github.com/docker/docker/pull/4572)
- requires docker to integrate user namespace support, and deciding what getpwnam() does for these uids.
- any features that help users avoid use of privileged containers (https://github.com/GoogleCloudPlatform/kubernetes/issues/391)

###project
K8s will have a have a `project` API object.

Initial Features:
- `project` object is immutable.
- `project` objects are statically populated in the K8s API store by reading a config file.  Only a K8s Cluster Admin can do this.
- In order to allow using `project` name as namespace for objects under that `project`, and to ensure the compound names are still DNS-compatible names, `project` names must be DNS label format.  

Improvements:
- have API calls to create and delete `project` objects.

Most API objects have an associated `project`:
- pods have a `project`.
- `project`s don't have a `project`, nor do `userAccount`s
   -  or else they belong to a special global `project`? 
- An object's `project` cannot be changed after creation.

In the Enterprise Profile:
   - a `userAccount` may have permission to access several `project`s.  API calls need to specify a `project`.  Client libs may fill this in using a stored preference.

In the Simple Profile:
   - There is a single default `project`.  No need to specify `project` when making an API call.

Project versus userAccount vs Labels:
- `userAccount`s are intended for audit logging, and to define who has access to `project`s.
- `labels` (see docs/labels.md) should be used to distinguish pods, users, and other objects that cooperate towards a common goal but are different in some way, such as version, or responsibilities.  
- `project`s provide a namespace for objects.


how is `project` selected when making a REST call?
- default to only project if there is only one.
- query parameter
- subdomain, e.g. http://myproject.kubernetes.example.com.  nginx proxy can translate that to a query parameter
  - Subdomains have potential scaling limits. If using project names to identify the domain, you have to defend against profane or vanity names, and probably support blacklist limits on new project names. In non-hosted environments is less of an issue. Requires that the apiservers be tied into DNS, which is onerous to configure in some environments. Nice in some contexts.
- offering a project API scope /projects/<id>/<kubernetes api>
- global access via globally unique id (where supported, which isn't very consistent today) /pods/<uuid>

## Authentication

Initially:
- Use bearer tokens to authenticate users.  
- Each API request must include a "Authorization: Bearer <token>" header.
- A token identifies a particular `userAccount`.  
- Administrator utility generates tokens at cluster setup.
- Tokens are long lived.
- Tokens are just long random strings base64 encoded.  No content.

Improvements:
- make it harder for the wrong party to use the token.  
  - shorter lived tokens
  - use tokens that are bound to the channel between the client and the api server

Things to consider for Improved implementation:
- JWT http://tools.ietf.org/html/draft-ietf-oauth-json-web-token-13
  - Possible library: http://code.google.com/p/goauth2/oauth/jwt
  - Use with endpoint that generates short-term bearer token.
- tokens that are bound to the channel between the client and the api server
     - http://www.ietf.org/proceedings/90/slides/slides-90-uta-0.pdf
     - http://www.browserauth.net

Where to do authentication.  Either:
- Authenticate in reverse proxy (currently nginx, could easily be apache+mod_auth).  Proxy either rejects invalid token or appends authorized_user identifier to the request before passing to APIserver.
- Apiserver checks token.

Considerations:
- In some arrangements, the proxy might terminate SSL, and use a CA-signed certificate, while the APIserver might just use a self-signed or organization-signed certificate.
- some large orgs will already have existing SaaS web services (e.g. storage, VMs).  Some will already have a gateway that handles auth for multiple services.  If K8s does auth in the proxy, then those orgs can just replace the proxy with their existing web service gateway.
- Apache or nginx is more stable than K8s code.  Prefer to put secrets (tokens, SSL cert.) in lower-touch place.
- Admins configuring K8s for enterprise use are more likely to know how to config a proxy than to modify Go code of apiserver. 

Based on above considerations, auth should happen in the proxy.


## Authorization

K8s authorization should:
- Allow for a range of maturity levels, from single-user for those test driving the system, to integration with existing to enterprise authorization systems.
- Allow for centralized management of users and policies.  In some organizations, this will mean that the definition of users and access policies needs to reside on a system other than k8s and encompass other web services (such as a storage service).
- Allow processes running in K8s Pods to take on identity, and to allow narrow scoping of permissions for those identities in order to limit damage from software faults.
- Have Authorization Policies exposed as API objects so that a single config file can create or delete Pods, Controllers, Services, and the identities and policies for those Pods and Controllers. 
- Be separate as much as practical from Authentication, to allow Authentication methods to change over time and space, without impacting Authorization policies.

K8s will implement a relatively simple 
[Attribute-Based Access Control][http://en.wikipedia.org/wiki/Attribute_Based_Access_Control] model.
The model will be described in more detail in a forthcoming document.  The model
- Be less complex than XACML
- Be easily recognizable to those familiar with Amazon IAM Policies.
- Have a subset/aliases/defaults which allow it to be used in a way comfortable to those users more familiar with Role-Based Access Control.

Authorization policy is set by creating a set of Policy objects.

The API Server will be the Enforcement Point for Policy. For each API call that it receives, it will construct the Attributes needed to evaluate the policy (what user is making the call, what resource they are accessing, what they are trying to do that resource, etc) and pass those attribytes to a Decision Point.  The Decision Point code evaluates the Attributes against all the Policies and allows or denys the API call.  The system will be modular enough that the Decision Point code can either be linked into the APIserver binary, or be another service that the apiserver calls for each Decision (with appropriate time-limited caching as needed for performance).

Policy objects may be applicable only to a single project; K8s Project Admins would be able to create those as needed.  Other Policy objects may be applicable to all projects; a K8s Cluster Admin might create those in order to authorize a new type of controller to be used by all projects, or to make a K8s User into a K8s Project Admin.)

## Accounting

The API should have a `quota` concept (see (https://github.com/GoogleCloudPlatform/kubernetes/issues/442
).  A quota object relates a project (and optionally a label selector) to a maximum quantity of resources that may be used (see resources.md).

Initially:
- a `quota` object is immutable.
- for hosted K8s systems that do billing, Project is recommended level for billing accounts.
- Every object that consumes resources should have a `project` so that Resource usage stats are roll-up-able to `project`. 
- K8s Cluster Admin sets quota objects by writing a config file.

Improvements:
- allow one project to charge the quota for one or more other projects.  This would be controlled by a policy which allows changing a billing_project= label on an object.
- allow quota to be set by project owners for (project x label) combinations (e.g. let "webserver" project use 100 cores, but to prevent accidents, don't allow "webserver" project and "instance=test" use more than 10 cores.
- tools to help write consistent quota config files based on number of minions, historical project usages, QoS needs, etc.
- way for K8s Cluster Admin to incrementally adjust Quota objects.

Simple profile:
   - a single `project` with infinite resource limits.

Enterprise profile:
   - multiple projects

Issues:
- need for locking or "eventual consistency" when multiple apiserver goroutines are accessing the object store and handling pod creations.


## Audit Logging

API actions can be logged.

Initial implementation:
- All API calls logged to nginx logs.

Improvements:
- API server does logging instead.
- Policies to drop logging for high rate trusted API calls, or by users performing audit or other sensitive functions.


