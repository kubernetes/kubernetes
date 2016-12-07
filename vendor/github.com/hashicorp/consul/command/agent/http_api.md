# Agent HTTP API

The Consul agent is capable of running an HTTP server that
exposes various API's in a RESTful manner. These API's can
be used to both query the service catalog, as well as to
register new services.

The URLs are also versioned to allow for changes in the API.
The current URLs supported are:

Catalog:
* /v1/catalog/register : Registers a new service
* /v1/catalog/deregister : Deregisters a service or node
* /v1/catalog/datacenters : Lists known datacenters
* /v1/catalog/nodes : Lists nodes in a given DC
* /v1/catalog/services : Lists services in a given DC
* /v1/catalog/service/<service>/ : Lists the nodes in a given service
* /v1/catalog/node/<node>/ : Lists the services provided by a node

Health system:
* /v1/health/node/<node>: Returns the health info of a node
* /v1/health/checks/<service>: Returns the checks of a service
* /v1/health/service/<service>: Returns the nodes and health info of a service
* /v1/health/state/<state>: Returns the checks in a given state

Status:
* /v1/status/leader : Returns the current Raft leader
* /v1/status/peers : Returns the current Raft peer set

Agent:
* /v1/agent/self : Returns the local configuration
* /v1/agent/checks : Returns the checks the local agent is managing
* /v1/agent/services : Returns the services local agent is managing
* /v1/agent/members : Returns the members as seen by the local serf agent
* /v1/agent/join/<node> : Instructs the local agent to join a node
* /v1/agent/force-leave/<node>: Instructs the agent to force a node into the left state
* /v1/agent/check/register
* /v1/agent/check/deregister/<name>
* /v1/agent/check/pass/<name>
* /v1/agent/check/warn/<name>
* /v1/agent/check/fail/<name>
* /v1/agent/service/register
* /v1/agent/service/deregister/<name>

KVS:
* /v1/kv/<key>

