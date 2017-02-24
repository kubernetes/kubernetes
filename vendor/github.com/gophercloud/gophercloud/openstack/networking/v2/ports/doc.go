// Package ports contains functionality for working with Neutron port resources.
// A port represents a virtual switch port on a logical network switch. Virtual
// instances attach their interfaces into ports. The logical port also defines
// the MAC address and the IP address(es) to be assigned to the interfaces
// plugged into them. When IP addresses are associated to a port, this also
// implies the port is associated with a subnet, as the IP address was taken
// from the allocation pool for a specific subnet.
package ports
