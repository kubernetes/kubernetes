# Design: Limit direct access to etcd from within Kubernetes

All nodes have effective access of "root" on the entire Kubernetes cluster today because they have access to etcd, the central data store.  The kubelet, the service proxy, and the nodes themselves have a connection to etcd that can be used to read or write any data in the system.  In a cluster with many hosts, any container or user that gains the ability to write to the network device that can reach etcd, on any host, also gains that access.

* The Kubelet and Kube Proxy currently rely on an efficient "wait for changes over HTTP" interface get their current state and avoid missing changes
  * This interface is implemented by etcd as the "watch" operation on a given key containing useful data


## Options:

1. Do nothing
2. Introduce an HTTP proxy that limits the ability of nodes to access etcd
    1. Prevent writes of data from the kubelet
    2. Prevent reading data not associated with the client responsibilities
    3. Introduce a security token granting access
3. Introduce an API on the apiserver that returns the data a node Kubelet and Kube Proxy needs
    1. Remove the ability of nodes to access etcd via network configuration
    2. Provide an alternate implementation for the event writing code Kubelet
    3. Implement efficient "watch for changes over HTTP" to offer comparable function with etcd
    4. Ensure that the apiserver can scale at or above the capacity of the etcd system.
    5. Implement authorization scoping for the nodes that limits the data they can view
4. Implement granular access control in etcd
    1. Authenticate HTTP clients with client certificates, tokens, or BASIC auth and authorize them for read only access
    2. Allow read access of certain subpaths based on what the requestor's tokens are


## Evaluation:

Option 1 would be considered unacceptable for deployment in a multi-tenant or security conscious environment.  It would be acceptable in a low security deployment where all software is trusted.  It would be acceptable in proof of concept environments on a single machine.

Option 2 would require implementing an http proxy that for 2-1 could block POST/PUT/DELETE requests (and potentially HTTP method tunneling parameters accepted by etcd).  2-2 would be more complicated and would require filtering operations based on deep understanding of the etcd API *and* the underlying schema.  It would be possible, but involve extra software.

Option 3 would involve extending the existing apiserver to return pods associated with a given node over an HTTP "watch for changes" mechanism, which is already implemented.  Proper security would involve checking that the caller is authorized to access that data - one imagines a per node token, key, or SSL certificate that could be used to authenticate and then authorize access to only the data belonging to that node.  The current event publishing mechanism from the kubelet would also need to be replaced with a secure API endpoint or a change to a polling model.  The apiserver would also need to be able to function in a horizontally scalable mode by changing or fixing the "operations" queue to work in a stateless, scalable model.  In practice, the amount of traffic even a large Kubernetes deployment would drive towards an apiserver would be tens of requests per second (500 hosts, 1 request per host every minute) which is negligible if well implemented.  Implementing this would also decouple the data store schema from the nodes, allowing a different data store technology to be added in the future without affecting existing nodes.  This would also expose that data to other consumers for their own purposes (monitoring, implementing service discovery).

Option 4 would involve extending etcd to [support access control](https://github.com/coreos/etcd/issues/91).  Administrators would need to authorize nodes to connect to etcd, and expose network routability directly to etcd.  The mechanism for handling this authentication and authorization would be different than the authorization used by Kubernetes controllers and API clients.  It would not be possible to completely replace etcd as a data store without also implementing a new Kubelet config endpoint.


## Preferred solution:

Implement the first parts of option 3 - an efficient watch API for the pod, service, and endpoints data for the Kubelet and Kube Proxy.  Authorization and authentication are planned in the future - when a solution is available, implement a custom authorization scope that allows API access to be restricted to only the data about a single node or the service endpoint data.

In general, option 4 is desirable in addition to option 3 as a mechanism to further secure the store to infrastructure components that must access it.


## Caveats

In all four options, compromise of a host will allow an attacker to imitate that host.  For attack vectors that are reproducible from inside containers (privilege escalation), an attacker can distribute himself to other hosts by requesting new containers be spun up.  In scenario 1, the cluster is totally compromised immediately.  In 2-1, the attacker can view all information about the cluster including keys or authorization data defined with pods.  In 2-2 and 3, the attacker must still distribute himself in order to get access to a large subset of information, and cannot see other data that is potentially located in etcd like side storage or system configuration.  For attack vectors that are not exploits, but instead allow network access to etcd, an attacker in 2ii has no ability to spread his influence, and is instead restricted to the subset of information on the host.  For 3-5, they can do nothing they could not do already (request access to the nodes / services endpoint) because the token is not visible to them on the host.

