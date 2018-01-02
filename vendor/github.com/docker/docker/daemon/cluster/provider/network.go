package provider

import "github.com/docker/docker/api/types"

// NetworkCreateRequest is a request when creating a network.
type NetworkCreateRequest struct {
	ID string
	types.NetworkCreateRequest
}

// NetworkCreateResponse is a response when creating a network.
type NetworkCreateResponse struct {
	ID string `json:"Id"`
}

// VirtualAddress represents a virtual address.
type VirtualAddress struct {
	IPv4 string
	IPv6 string
}

// PortConfig represents a port configuration.
type PortConfig struct {
	Name          string
	Protocol      int32
	TargetPort    uint32
	PublishedPort uint32
}

// ServiceConfig represents a service configuration.
type ServiceConfig struct {
	ID               string
	Name             string
	Aliases          map[string][]string
	VirtualAddresses map[string]*VirtualAddress
	ExposedPorts     []*PortConfig
}
