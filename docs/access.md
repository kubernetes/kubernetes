# K8s Identity and Access Management Sketch

This document suggests a direction for identity and access management in the Kubernetes system. 


## Background

High level goals are:
   - Have a plan for how identity, authentication, and authorization will fit in to the API.
   - Have a plan for partitioning resources within a cluster between independent organizational units.
   - Ease integration with existing enterprise and hosted scenarios.

### Actors
Each of these can act as normal users or attackers.
   - End users (e.g. visitors to a website which is served by programs which are run on kubernetes)
   - k8s users (e.g. people who create k8s API objects like Pods)
   - k8s insiders (e.g. people who setup k8s binaries, or others with special access to machines and networks.)

### Threats
Both intentional attacks and accidental use of privilege are concerns.

For both cases it may be useful to think about these categories differently:
  - Application Path - attack by sending network messages from the internet to the IP/port of any application running on k8s.  May exploit weakness in application or misconfiguration of k8s.
  - K8s API Path - attack by sending network messages to any k8s infrastructure network endpoint.  Attacker may have k8s user identity.
  - Insider Path - attack on k8s system components.  Attacker may have privileged access to networks, machines or k8s software and data.  Software errors in k8s system components and administrator error are some types of threat in this category.

This document is primarily concerned with K8s API paths, and secondarily with Internal paths.   The Application path also needs to be secure, but is not the focus of this document.
 
### Assets to protect

End user assets:
   - Personal information like private messages, or images uploaded by end users
   - web server logs

K8s user assets:
  - end-user assets of each k8s user
  - things private to the K8s app, like:
    - credentials for accessing other services (docker private repos, storage services, facebook, etc)
    - SSL certificates for web servers
    - proprietary data and code

K8s Cluster assets:
  - Assets of each k8s-user
  - certificates which identify the k8s machines and servers to users.
  - cluster resources (the value of k8s computing resources).

This document is primarily about protecting k8s user assets and K8s cluster assets from other k8s users and k8s insiders.
 
### Usage environments
Cluster in Small organization:  
   - k8s insiders may be the same people as k8s users.  
   - few k8s insiders.  
    - prefer ease of use to fine-grained access control/precise accounting, etc.
 - Product requirement that it be easy for potential users to try out setting up a simple cluster.

Cluster in Large organization: 
   - k8s insiders typically distinct people from k8s users.  may need to divide k8s insider access by roles. 
   - k8s users need to be protected from each other.  
   - auditing of k8s user and k8s insider actions important.   
   - flexible accurate usage accounting and resource controls important.
   - Lots of automated access to APIs.  
   - Need to integrate with existing enterprise directory, authentication, accounting, auditing, and security policy infrastructure.

Org-run cluster: 
   - organization that runs k8s master components is same as the org that runs apps on k8s.  
   - Minions may be on-premises VMs or physical machines; Cloud VMs; or a mix.
 
Hosted cluster:
  - Offering K8s API as a service, or offering a Paas or Saas built on K8s
  - May already offer web services, and need to integrate with existing customer account concept, and existing authentication, accounting, auditing, and security policy infrastructure.
  - May want to leverage K8s user accounts and accounting to manage their user accounts (not a priority to support this use case.)
  - Precise and accurate accounting of resources needed.

K8s ecosystem services:
 - There may be companies that want to offer their existing services (Build, CI, A/B-test, release automation, etc) for use with K8s.  There should be some story for this case.

Pods configs should be largely portable between Org-run and hosted configurations.  


# Design 
Related discussion: https://github.com/GoogleCloudPlatform/kubernetes/issues/443

This doc describes two security profiles:
  - Simple profile:  like single-user mode.  Make it easy to evaluate k8s without lots of configuring accounts and policies.  Protects from unauthorized users, but does not partition authorized users.
  - Enterprise profile: Provide mechanisms needed for large numbers of users.  Defence in depth.  Should integrate with existing enterprise security infrastructure.
 
K8s distribution should include templates of config, and documentation, for simple and enterprise profiles.  System should be flexible enough for knowledgeable users to create intermediate profiles, but K8s developers should only reason about those two Profiles, not a matrix.

Features in this doc are divided into "Initial Feature", and "Improvements".   Initial features would be candidates for version 1.00.

## Identity
###userAccount
K8s will have a `userAccount` API object.
- `userAccount` has a name which is a string. 
- `userAccount` is not the same as the unix username of processes 
- 'userAccount` objects have a name.  The name should allow relatively free-form strings (at least flexible enough to hold most email addresses) to aid integration with existing enterprise auth systems, or existing account identifiers in hosting systems.
- `userAccount` API objects can have labels

Initial Features:
- `userAccount` object is immutable
- there is no superuser `userAccount`
- `userAccount` objects are statically populated in the k8s API store by reading a config file.  Only a k8s admin can do this.
- `userAccount` can have a default `project`.  If API call does not specify a `project`, the default `project` for that caller is assumed.
- `userAccount` may access multiple projects.

Improvements:
- Have API calls to create and delete `userAccount`s.
- Make `userAccount` part of a separate API group from core k8s objects like `pod`.  Facilitates plugging in alternate Access Management.

Simple Profile:
   - single `userAccount`, used by all users.

Enterprise Profile:
   - every human user has own `userAccount`. 
   - `userAccount`s have labels that indicate both membership in groups, and ability to act in certain roles.
   - each service using the API has own `userAccount` too. (e.g. `scheduler`, `repcontroller`)
   - list of users may be synced in from existing enterprise directory service.
      - issue: deleting username still being referenced by pod, etc.

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
- `project` objects are statically populated in the k8s API store by reading a config file.  Only admin can do this.
- In order to allow using `project` name as namespace for objects under that `project`, and to ensure the compound names are still DNS-compatible names, `project` names must be DNS label format.  

Improvements:
- have API calls to create and delete `project` objects.
- sync in from enterprise directory service.

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
- query parameter
- default to only project if there is only one.
- subdomain, e.g. http://myproject.kubernetes.example.com.  nginx proxy can translate that to a query parameter

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
     - http://ww.browserauth.net

Where to do authentication.  Either:
- Authenticate in reverse proxy (nginx).  Proxy either rejects invalid token or appends authorized_user identifier to the request before passing to APIserver.
- Apiserver checks token.

Considerations:
- In some arrangements, the proxy might terminate SSL, and use a CA-signed certificate, while the APIserver might just use a self-signed or organization-signed certificate.
- some large orgs will already have existing SaaS web services (e.g. storage, VMs).  Some will already have a gateway that handles auth for multiple services.  If K8s does auth in the proxy, then those orgs can just replace the proxy with their existing web service gateway.
- nginx is more stable than k8s code.  Prefer to put secrets (tokens, SSL cert.) in lower-touch place.
- Admins configuring k8s for enterprise use are more likely to know how to config a proxy than to modify Go code of apiserver. 

Based on above considerations, auth should happen in the proxy.

Implementation options:
- Simplest: a list of encoded tokens is stored in nginx config file as a map.  a rewrite rule (http://nginx.org/en/docs/http/ngx_http_rewrite_module.html) adds authorized_user if Auth header is in the is in the map, else returns a 401 Unauthorized.
- nginx can defer auth to a separate auth server.  This could aid integration with existing enterprise auth solutions.  e.g.:
      - https://github.com/algermissen/nginx-dlg-auth/blob/master/README.md
      - http://nginx.org/en/docs/http/ngx_http_auth_request_module.html
- nginx can have lua plugin:  https://github.com/openresty/lua-nginx-module 
- User can provide own proxy, other than nginx.


## Authorization

Authorization is done by the APIserver.

Authorization is done using Policy objects.


### Policy objects
Policy objects are API objects.  They express http://en.wikipedia.org/wiki/Attribute_Based_Access_Control 

Simple Profile:
- one Policy object that allows the single `userAccount` to CRUD objects in the single `project`.

Enterprise Profile:
- Many policy objects in each of many projects.
- Tools and services that wrap policy creation interface to enforce meta policies, do template expansions, report, etc.



Initial Features:
- Policy object is immutable
- Policy objects are statically populated in the k8s API store by reading a config file.  Only admin can do this.
- Just a few policies per `project` which list which users can create objects, which can just view, them, etc.
- Objects are created with reference to these default policies.

Improvements:
- Have API calls to create and delete and modify Policy objects.   These would be in a separate API group from core k8s APIs.  This allows for replacing the k8s authorization service with an alternate implementation, and to centralize policies that might apply to services other than k8s.
- Ability to change policy for an object.
- Ability to create an object with a non-default policy effective immediately at creation time.
- Ability to defer policy object checking to a policy server.
- Authorization tokens to authorize entities without a `userAccount`.


### Policy object format

Policy Object:
```go
type policies map[string]policy

type policy {
    project string, // ref to project of this Policy, to namespace the name.  
    name string, // name within the project name of the Policy
    a PolicyType,
    s Subject,
    v Verb,
    o Object 
    expires string // RFC3339
}
type PolicyType string
const {
   ALLOW PolicyType = "ALLOW",
   // Later DENY, etc.
}

type Subject string // Serialized label selector
type Verb string
const {
   GET Verb = "GET",
   // ... other HTTP methods.
   ANY Verb = "ANY" // any http method
   CREATE Verb = "CREATE" // PUT or POST
   // ...
}
type Object {
  exact string  // any resource with exactly this path
  // OR
  prefix string // any resource with this path prefix (after removing "/api/<version>/")
  // OR
  where string // serialized label selector.
}
```

Ideally, lists of policy objects would have relatively concise and readable YAML forms, such as:
```
{name: bob_can_read_pods, a: ALLOW, s: user.name is bob@example.com, v: GET, prefix: /pods}
{name: admins_can_delete_pods, a: ALLOW, s: user.role is admin, v: DELETE, prefix: /pods}
{name: tmp1234, a: ALLOW, s: user.name is "some.agent", v: POST, prefix: /pods/somepod, expires: 2014-08-13 16:21:42-07:00 }
```

Requests that don't match at least one ALLOW are not allowed.
TODO: define DENY, and other operations and their precedence.

Delegation can be implemented by writing new narrowly tailored policies.
TODO: example of policy to delegate pod creation from a podTemplate (see https://github.com/GoogleCloudPlatform/kubernetes/issues/170).

### Architecture for Authorization
When the APIserver receives a new request, it passes the
the `userAccount`, http method, and http path to an `Authorize() method`.

In a simple implementation, the Authorize() module:
  - runs in the APIserver
  - searches all policy objects for a match.
  - updates its cache  when new Policy is added.

In alternate implementations, it may:
  - have indexes to speed matching.  (Maybe this can share code with Label Queries.)
  - defer to a centralized auth server for the enterprise. 


### Labels
Initially, IIUC, labels are strings and not API objects in their own right. 
Eventually, labels may have policies or namespaces which restrict application of certain labels.
  

## Accounting

The API should have a `quota` concept (see (https://github.com/GoogleCloudPlatform/kubernetes/issues/442
).  A quota object relates a project (and optionally a label selector) to a maximum quantity of resources that may be used (see resources.md).

Initially:
- a `quota` object is immutable.
- for hosted k8s systems that do billing, Project is recommended level for billing accounts.
- Every object that consumes resources should have a `project` so that Resource usage stats are roll-up-able to `project`. 
- admin sets quota objects by writing a config file.

Improvements:
- allow one project to charge the quota for one or more other projects.  This would be controlled by a policy which allows changing a billing_project= label on an object.
- allow quota to be set by project owners for (project x label) combinations (e.g. let "webserver" project use 100 cores, but to prevent accidents, don't allow "webserver" project and "instance=test" use more than 10 cores.
- tools to help write consistent quota config files based on number of minions, historical project usages, QoS needs, etc.
- way for admin to incrementally adjust Quota objects.

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


