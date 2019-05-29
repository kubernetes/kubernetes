package swarm // import "github.com/docker/docker/api/types/swarm"

import (
	"github.com/docker/docker/api/types/network"
)

// Endpoint represents an endpoint.
type Endpoint struct {
	Spec       EndpointSpec        `json:",omitempty"`
	Ports      []PortConfig        `json:",omitempty"`
	VirtualIPs []EndpointVirtualIP `json:",omitempty"`
}

// EndpointSpec represents the spec of an endpoint.
type EndpointSpec struct {
	Mode  ResolutionMode `json:",omitempty"`
	Ports []PortConfig   `json:",omitempty"`
}

// ResolutionMode represents a resolution mode.
type ResolutionMode string

const (
	// ResolutionModeVIP VIP
	ResolutionModeVIP ResolutionMode = "vip"
	// ResolutionModeDNSRR DNSRR
	ResolutionModeDNSRR ResolutionMode = "dnsrr"
)

// PortConfig represents the config of a port.
type PortConfig struct {
	Name     string             `json:",omitempty"`
	Protocol PortConfigProtocol `json:",omitempty"`
	// TargetPort is the port inside the container
	TargetPort uint32 `json:",omitempty"`
	// PublishedPort is the port on the swarm hosts
	PublishedPort uint32 `json:",omitempty"`
	// PublishMode is the mode in which port is published
	PublishMode PortConfigPublishMode `json:",omitempty"`
}

// PortConfigPublishMode represents the mode in which the port is to
// be published.
type PortConfigPublishMode string

const (
	// PortConfigPublishModeIngress is used for ports published
	// for ingress load balancing using routing mesh.
	PortConfigPublishModeIngress PortConfigPublishMode = "ingress"
	// PortConfigPublishModeHost is used for ports published
	// for direct host level access on the host where the task is running.
	PortConfigPublishModeHost PortConfigPublishMode = "host"
)

// PortConfigProtocol represents the protocol of a port.
type PortConfigProtocol string

const (
	// TODO(stevvooe): These should be used generally, not just for PortConfig.

	// PortConfigProtocolTCP TCP
	PortConfigProtocolTCP PortConfigProtocol = "tcp"
	// PortConfigProtocolUDP UDP
	PortConfigProtocolUDP PortConfigProtocol = "udp"
	// PortConfigProtocolSCTP SCTP
	PortConfigProtocolSCTP PortConfigProtocol = "sctp"
)

// EndpointVirtualIP represents the virtual ip of a port.
type EndpointVirtualIP struct {
	NetworkID string `json:",omitempty"`
	Addr      string `json:",omitempty"`
}

// Network represents a network.
type Network struct {
	ID string
	Meta
	Spec        NetworkSpec  `json:",omitempty"`
	DriverState Driver       `json:",omitempty"`
	IPAMOptions *IPAMOptions `json:",omitempty"`
}

// NetworkSpec represents the spec of a network.
type NetworkSpec struct {
	Annotations
	DriverConfiguration *Driver                  `json:",omitempty"`
	IPv6Enabled         bool                     `json:",omitempty"`
	Internal            bool                     `json:",omitempty"`
	Attachable          bool                     `json:",omitempty"`
	Ingress             bool                     `json:",omitempty"`
	IPAMOptions         *IPAMOptions             `json:",omitempty"`
	ConfigFrom          *network.ConfigReference `json:",omitempty"`
	Scope               string                   `json:",omitempty"`
}

// NetworkAttachmentConfig represents the configuration of a network attachment.
type NetworkAttachmentConfig struct {
	Target     string            `json:",omitempty"`
	Aliases    []string          `json:",omitempty"`
	DriverOpts map[string]string `json:",omitempty"`
}

// NetworkAttachment represents a network attachment.
type NetworkAttachment struct {
	Network   Network  `json:",omitempty"`
	Addresses []string `json:",omitempty"`
}

// IPAMOptions represents ipam options.
type IPAMOptions struct {
	Driver  Driver       `json:",omitempty"`
	Configs []IPAMConfig `json:",omitempty"`
}

// IPAMConfig represents ipam configuration.
type IPAMConfig struct {
	Subnet  string `json:",omitempty"`
	Range   string `json:",omitempty"`
	Gateway string `json:",omitempty"`
}
