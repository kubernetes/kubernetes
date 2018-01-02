package network

import (
	networktypes "github.com/docker/docker/api/types/network"
	clustertypes "github.com/docker/docker/daemon/cluster/provider"
	"github.com/docker/go-connections/nat"
)

// Settings stores configuration details about the daemon network config
// TODO Windows. Many of these fields can be factored out.,
type Settings struct {
	Bridge                 string
	SandboxID              string
	HairpinMode            bool
	LinkLocalIPv6Address   string
	LinkLocalIPv6PrefixLen int
	Networks               map[string]*EndpointSettings
	Service                *clustertypes.ServiceConfig
	Ports                  nat.PortMap
	SandboxKey             string
	SecondaryIPAddresses   []networktypes.Address
	SecondaryIPv6Addresses []networktypes.Address
	IsAnonymousEndpoint    bool
	HasSwarmEndpoint       bool
}

// EndpointSettings is a package local wrapper for
// networktypes.EndpointSettings which stores Endpoint state that
// needs to be persisted to disk but not exposed in the api.
type EndpointSettings struct {
	*networktypes.EndpointSettings
	IPAMOperational bool
}
