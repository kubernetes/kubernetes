/*
Package acl provides information and interaction with the access lists feature
of the Rackspace Cloud Load Balancer service.

The access list management feature allows fine-grained network access controls
to be applied to the load balancer's virtual IP address. A single IP address,
multiple IP addresses, or entire network subnets can be added. Items that are
configured with the ALLOW type always takes precedence over items with the DENY
type. To reject traffic from all items except for those with the ALLOW type,
add a networkItem with an address of "0.0.0.0/0" and a DENY type.
*/
package acl
