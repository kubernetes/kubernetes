package dns

// PortDNSExt represents a decorated form of a Port with the additional
// Port DNS information.
type PortDNSExt struct {
	// The DNS name of the port.
	DNSName string `json:"dns_name"`

	// The DNS assignment of the port.
	DNSAssignment []map[string]string `json:"dns_assignment"`
}

// FloatingIPDNSExt represents a decorated form of a Floating IP with the
// additional Floating IP DNS information.
type FloatingIPDNSExt struct {
	// The DNS name of the floating IP, assigned to the external DNS
	// service.
	DNSName string `json:"dns_name"`

	// The DNS domain of the floating IP, assigned to the external DNS
	// service.
	DNSDomain string `json:"dns_domain"`
}

// NetworkDNSExt represents a decorated form of a Network with the additional
// Network DNS information.
type NetworkDNSExt struct {
	// The DNS domain of the network.
	DNSDomain string `json:"dns_domain"`
}
