package attachinterfaces

import (
	"github.com/gophercloud/gophercloud/pagination"
)

// FixedIP represents a Fixed IP Address.
type FixedIP struct {
	SubnetID  string `json:"subnet_id"`
	IPAddress string `json:"ip_address"`
}

// Interface represents a network interface on an instance.
type Interface struct {
	PortState string    `json:"port_state"`
	FixedIPs  []FixedIP `json:"fixed_ips"`
	PortID    string    `json:"port_id"`
	NetID     string    `json:"net_id"`
	MACAddr   string    `json:"mac_addr"`
}

// InterfacePage abstracts the raw results of making a List() request against the API.
// As OpenStack extensions may freely alter the response bodies of structures returned
// to the client, you may only safely access the data provided through the ExtractInterfaces call.
type InterfacePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if an InterfacePage contains no interfaces.
func (r InterfacePage) IsEmpty() (bool, error) {
	interfaces, err := ExtractInterfaces(r)
	return len(interfaces) == 0, err
}

// ExtractInterfaces interprets the results of a single page from a List() call,
// producing a map of interfaces.
func ExtractInterfaces(r pagination.Page) ([]Interface, error) {
	var s struct {
		Interfaces []Interface `json:"interfaceAttachments"`
	}
	err := (r.(InterfacePage)).ExtractInto(&s)
	return s.Interfaces, err
}
