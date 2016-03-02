/*
Package nodes provides information and interaction with the Node API resource
for the Rackspace Cloud Load Balancer service.

Nodes are responsible for servicing the requests received through the load
balancer's virtual IP. A node is usually a virtual machine. By default, the
load balancer employs a basic health check that ensures the node is listening
on its defined port. The node is checked at the time of addition and at regular
intervals as defined by the load balancer's health check configuration. If a
back-end node is not listening on its port, or does not meet the conditions of
the defined check, then connections will not be forwarded to the node, and its
status is changed to OFFLINE. Only nodes that are in an ONLINE status receive
and can service traffic from the load balancer.

All nodes have an associated status that indicates whether the node is
ONLINE, OFFLINE, or DRAINING. Only nodes that are in ONLINE status can receive
and service traffic from the load balancer. The OFFLINE status represents a
node that cannot accept or service traffic. A node in DRAINING status
represents a node that stops the traffic manager from sending any additional
new connections to the node, but honors established sessions. If the traffic
manager receives a request and session persistence requires that the node is
used, the traffic manager uses it. The status is determined by the passive or
active health monitors.

If the WEIGHTED_ROUND_ROBIN load balancer algorithm mode is selected, then the
caller should assign the relevant weights to the node as part of the weight
attribute of the node element. When the algorithm of the load balancer is
changed to WEIGHTED_ROUND_ROBIN and the nodes do not already have an assigned
weight, the service automatically sets the weight to 1 for all nodes.

One or more secondary nodes can be added to a specified load balancer so that
if all the primary nodes fail, traffic can be redirected to secondary nodes.
The type attribute allows configuring the node as either PRIMARY or SECONDARY.
*/
package nodes
