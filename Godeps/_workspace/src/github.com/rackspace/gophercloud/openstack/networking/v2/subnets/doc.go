// Package subnets contains functionality for working with Neutron subnet
// resources. A subnet represents an IP address block that can be used to
// assign IP addresses to virtual instances. Each subnet must have a CIDR and
// must be associated with a network. IPs can either be selected from the whole
// subnet CIDR or from allocation pools specified by the user.
//
// A subnet can also have a gateway, a list of DNS name servers, and host routes.
// This information is pushed to instances whose interfaces are associated with
// the subnet.
package subnets
