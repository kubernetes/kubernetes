package netlink

import (
	"fmt"
	"net"
)

// Neigh represents a link layer neighbor from netlink.
type Neigh struct {
	LinkIndex    int
	Family       int
	State        int
	Type         int
	Flags        int
	IP           net.IP
	HardwareAddr net.HardwareAddr
	LLIPAddr     net.IP //Used in the case of NHRP
	Vlan         int
	VNI          int
	MasterIndex  int
}

// String returns $ip/$hwaddr $label
func (neigh *Neigh) String() string {
	return fmt.Sprintf("%s %s", neigh.IP, neigh.HardwareAddr)
}

// NeighUpdate is sent when a neighbor changes - type is RTM_NEWNEIGH or RTM_DELNEIGH.
type NeighUpdate struct {
	Type uint16
	Neigh
}
