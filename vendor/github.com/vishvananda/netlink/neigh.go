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
	FlagsExt     int
	IP           net.IP
	HardwareAddr net.HardwareAddr
	LLIPAddr     net.IP //Used in the case of NHRP
	Vlan         int
	VNI          int
	MasterIndex  int

	// These values are expressed as "clock ticks ago".  To
	// convert these clock ticks to seconds divide by sysconf(_SC_CLK_TCK).
	// When _SC_CLK_TCK is 100, for example, the ndm_* times are expressed
	// in centiseconds.
	Confirmed uint32 // The last time ARP/ND succeeded OR higher layer confirmation was received
	Used      uint32 // The last time ARP/ND took place for this neighbor
	Updated   uint32 // The time when the current NUD state was entered
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
