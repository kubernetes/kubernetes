# Federated API Servers

## Abstract

We want to divide the single monolithic API server into multiple federated
servers. Anyone should be able to write their own federated API server to expose APIs they want.
Cluster admins should be able to expose new APIs at runtime by bringing up new
federated servers.

## Motivation

* Extensibility: We want to allow community members to write their own API
  servers to expose APIs they want. Cluster admins should be able to use these
  servers without having to require any change in the core kubernetes
  repository.
* Unblock new APIs from core kubernetes team review: A lot of new API proposals
  are currently blocked on review from the core kubernetes team. By allowing
  developers to expose their APIs as a separate server and enabling the cluster
  admin to use it without any change to the core kubernetes repository, we
  unblock these APIs.
* Place for staging experimental APIs: New APIs can remain in separate
  federated servers until they become stable, at which point, they can be moved
  to the core kubernetes master, if appropriate.
* Ensure that new APIs follow kubernetes conventions: Without the mechanism
  proposed here, community members might be forced to roll their own thing which
  may or may not follow kubernetes conventions.

## Goal

* Developers should be able to write their own API server and cluster admins
  should be able to add them to their cluster, exposing new APIs at runtime. All
  of this should not require any change to the core kubernetes API server.
* These new APIs should be seamless extension of the core kubernetes APIs (ex:
  they should be operated upon via kubectl).

## Non Goals

The following are related but are not the goals of this specific proposal:
* Make it easy to write a kubernetes API server.

## High Level Architecture

There will be 2 new components in the cluster:
* A simple program to summarize discovery information from all the servers.
* A reverse proxy to proxy client requests to individual servers.

The reverse proxy is optional. Clients can discover server URLs using the
summarized discovery information and contact them directly. Simple clients, can
always use the proxy.
The same program can provide both discovery summarization and reverse proxy.

### Constraints

* Unique API groups across servers: Each API server (and groups of servers, in HA)
  should expose unique API groups.
* Follow API conventions: APIs exposed by every API server should adhere to [kubernetes API
  conventions](../devel/api-conventions.md).
* Support discovery API: Each API server should support the kubernetes discovery API
  (list the suported groupVersions at `/apis` and list the supported resources
  at `/apis/<groupVersion>/`)
* No bootstrap problem: The core kubernetes server should not depend on any
  other federated server to come up. Other servers can only depend on the core
  kubernetes server.

## Implementation Details

### Summarizing discovery information

We can have a very simple Go program to summarize discovery information from all
servers. Cluster admins will register each federated API server (its baseURL and swagger
spec path) with the proxy. The proxy will summarize the list of all group versions
exposed by all registered API servers with their individual URLs at `/apis`.

### Reverse proxy

We can use any standard reverse proxy server like nginx or extend the same Go program that
summarizes discovery information to act as reverse proxy for all federated servers.

Cluster admins are also free to use any of the multiple open source API management tools
(for example, there is [Kong](https://getkong.org/), which is written in lua and there is
[Tyk](https://tyk.io/), which is written in Go). These API management tools
provide a lot more functionality like: rate-limiting, caching, logging,
transformations and authentication.
In future, we can also use ingress. That will give cluster admins the flexibility to
easily swap out the ingress controller by a Go reverse proxy, nginx, haproxy
or any other solution they might want.

### Storage

Each API server is responsible for storing their resources. They can have their
own etcd or can use kubernetes server's etcd using [third party
resources](../design/extending-api.md#adding-custom-resources-to-the-kubernetes-api-server).

### Health check

Kubernetes server's `/api/v1/componentstatuses` will continue to report status
of master components that it depends on (scheduler and various controllers).
Since clients have access to server URLs, they can use that to do
health check of individual servers.
In future, if a global health check is required, we can expose a health check
endpoint in the proxy that will report the status of all federated api servers
in the cluster.

### Auth

Since the actual server which serves client's request can be opaque to the client,
all API servers need to have homogeneous authentication and authorisation mechanisms.
All API servers will handle authn and authz for their resources themselves.
In future, we can also have the proxy do the auth and then have apiservers trust
it (via client certs) to report the actual user in an X-something header.

For now, we will trust system admins to configure homogeneous auth on all servers.
Future proposals will refine how auth is managed across the cluster.

### kubectl

kubectl will talk to the discovery endpoint (or proxy) and use the discovery API to
figure out the operations and resources supported in the cluster.
Today, it uses RESTMapper to determine that. We will update kubectl code to populate
RESTMapper using the discovery API so that we can add and remove resources
at runtime.
We will also need to make kubectl truly generic. Right now, a lot of operations
(like get, describe) are hardcoded in the binary for all resources. A future
proposal will provide details on moving those operations to server.

Note that it is possible for kubectl to talk to individual servers directly in
which case proxy will not be required at all, but this requires a bit more logic
in kubectl. We can do this in future, if desired.

### Handling global policies

Now that we have resources spread across multiple API servers, we need to
be careful to ensure that global policies (limit ranges, resource quotas, etc) are enforced.
Future proposals will improve how this is done across the cluster.

#### Namespaces

When a namespaced resource is created in any of the federated server, that
server first needs to check with the kubernetes server that:

* The namespace exists.
* User has authorization to create resources in that namespace.
* Resource quota for the namespace is not exceeded.

To prevent race conditions, the kubernetes server might need to expose an atomic
API for all these operations.

While deleting a namespace, kubernetes server needs to ensure that resources in
that namespace maintained by other servers are deleted as well. We can do this
using resource [finalizers](../design/namespaces.md#finalizers). Each server
will add themselves in the set of finalizers before they create a resource in
the corresponding namespace and delete all their resources in that namespace,
whenever it is to be deleted (kubernetes API server already has this code, we
will refactor it into a library to enable reuse).

Future proposal will talk about this in more detail and provide a better
mechanism.

#### Limit ranges and resource quotas

kubernetes server maintains [resource quotas](../admin/resourcequota/README.md) and
[limit ranges](../admin/limitrange/README.md) for all resources.
Federated servers will need to check with the kubernetes server before creating any
resource.

## Running on hosted kubernetes cluster

This proposal is not enough for hosted cluster users, but allows us to improve
that in the future.
On a hosted kubernetes cluster, for e.g. on GKE - where Google manages the kubernetes
API server, users will have to bring up and maintain the proxy and federated servers
themselves.
Other system components like the various controllers, will not be aware of the
proxy and will only talk to the kubernetes API server.

One possible solution to fix this is to update kubernetes API server to detect when
there are federated servers in the cluster and then change its advertise address to
the IP address of the proxy.
Future proposal will talk about this in more detail.

## Alternatives

There were other alternatives that we had discussed.

* Instead of adding a proxy in front, let the core kubernetes server provide an
  API for other servers to register themselves. It can also provide a discovery
  API which the clients can use to discover other servers and then talk to them
  directly. But this would have required another server API a lot of client logic as well.
* Validating federated servers: We can validate new servers when they are registered
  with the proxy, or keep validating them at regular intervals, or validate
  them only when explicitly requested, or not validate at all.
  We decided that the proxy will just assume that all the servers are valid
  (conform to our api conventions). In future, we can provide conformance tests.

## Future Work

* Validate servers: We should have some conformance tests that validate that the
  servers follow kubernetes api-conventions.
* Provide centralised auth service: It is very hard to ensure homogeneous auth
  across multiple federated servers, especially in case of hosted clusters
  (where different people control the different servers). We can fix it by
  providing a centralised authentication and authorization service which all of
  the servers can use.



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/federated-api-servers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
