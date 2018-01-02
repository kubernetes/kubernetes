package cluster

import (
	"github.com/docker/docker/api/types/network"
	"golang.org/x/net/context"
)

const (
	// EventSocketChange control socket changed
	EventSocketChange = iota
	// EventNodeReady cluster node in ready state
	EventNodeReady
	// EventNodeLeave node is leaving the cluster
	EventNodeLeave
	// EventNetworkKeysAvailable network keys correctly configured in the networking layer
	EventNetworkKeysAvailable
)

// ConfigEventType type of the event produced by the cluster
type ConfigEventType uint8

// Provider provides clustering config details
type Provider interface {
	IsManager() bool
	IsAgent() bool
	GetLocalAddress() string
	GetListenAddress() string
	GetAdvertiseAddress() string
	GetDataPathAddress() string
	GetRemoteAddressList() []string
	ListenClusterEvents() <-chan ConfigEventType
	AttachNetwork(string, string, []string) (*network.NetworkingConfig, error)
	DetachNetwork(string, string) error
	UpdateAttachment(string, string, *network.NetworkingConfig) error
	WaitForDetachment(context.Context, string, string, string, string) error
}
