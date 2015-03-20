/*
Package monitors provides information and interaction with the Health Monitor
API resource for the Rackspace Cloud Load Balancer service.

The load balancing service includes a health monitoring resource that
periodically checks your back-end nodes to ensure they are responding correctly.
If a node does not respond, it is removed from rotation until the health monitor
determines that the node is functional. In addition to being performed
periodically, a health check also executes against every new node that is
added, to ensure that the node is operating properly before allowing it to
service traffic. Only one health monitor is allowed to be enabled on a load
balancer at a time.

As part of a good strategy for monitoring connections, secondary nodes should
also be created which provide failover for effectively routing traffic in case
the primary node fails. This is an additional feature that ensures that you
remain up in case your primary node fails.

There are three types of health monitor: CONNECT, HTTP and HTTPS.
*/
package monitors
