# Consul RPC Endpoints

Consul provides a few high-level services, each of which exposes
methods. The services exposed are:

* Status : Used to query status information
* Catalog: Used to register, deregister, and query service information
* Health: Used to notify of health checks and changes to health

## Status Service

The status service is used to query for various status information
from the Consul service. It exposes the following methods:

* Ping : Used to test connectivity
* Leader : Used to get the address of the leader
* Peers: Used to get the Raft peerset

## Catalog Service

The catalog service is used to manage service discovery and registration.
Nodes can register the services they provide, and deregister them later.
The service exposes the following methods:

* Register : Registers a node, and potentially a node service and check
* Deregister : Deregisters a node, and potentially a node service or check

* ListDatacenters: List the known datacenters
* ListServices : Lists the available services
* ListNodes : Lists the available nodes
* ServiceNodes: Returns the nodes that are part of a service
* NodeServices: Returns the services that a node is registered for

## Health Service

The health service is used to manage health checking. Nodes have system
health checks, as well as application health checks. This service is used to
query health information, as well as for nodes to publish changes.

* ChecksInState : Gets the checks that in a given state
* NodeChecks: Gets the checks a given node has
* ServiceChecks: Gets the checks a given service has
* ServiceNodes: Returns the nodes that are part of a service, including health info

