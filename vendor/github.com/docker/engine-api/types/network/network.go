package network

// Address represents an IP address
type Address struct {
	Addr      string
	PrefixLen int
}

// IPAM represents IP Address Management
type IPAM struct {
	Driver  string
	Options map[string]string //Per network IPAM driver options
	Config  []IPAMConfig
}

// IPAMConfig represents IPAM configurations
type IPAMConfig struct {
	Subnet     string            `json:",omitempty"`
	IPRange    string            `json:",omitempty"`
	Gateway    string            `json:",omitempty"`
	AuxAddress map[string]string `json:"AuxiliaryAddresses,omitempty"`
}

// EndpointIPAMConfig represents IPAM configurations for the endpoint
type EndpointIPAMConfig struct {
	IPv4Address  string   `json:",omitempty"`
	IPv6Address  string   `json:",omitempty"`
	LinkLocalIPs []string `json:",omitempty"`
}

// EndpointSettings stores the network endpoint details
type EndpointSettings struct {
	// Configurations
	IPAMConfig *EndpointIPAMConfig
	Links      []string
	Aliases    []string
	// Operational data
	NetworkID           string
	EndpointID          string
	Gateway             string
	IPAddress           string
	IPPrefixLen         int
	IPv6Gateway         string
	GlobalIPv6Address   string
	GlobalIPv6PrefixLen int
	MacAddress          string
}

// NetworkingConfig represents the container's networking configuration for each of its interfaces
// Carries the networking configs specified in the `docker run` and `docker network connect` commands
type NetworkingConfig struct {
	EndpointsConfig map[string]*EndpointSettings // Endpoint configs for each connecting network
}
