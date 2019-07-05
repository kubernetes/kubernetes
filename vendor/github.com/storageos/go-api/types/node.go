package types

import (
	"time"
)

// Node represents a StorageOS cluster node.
type Node struct {
	NodeConfig

	HostID      uint32    `json:"hostID"`
	Name        string    `json:"name"`
	Description string    `json:"description"`
	CreatedAt   time.Time `json:"createdAt"`
	UpdatedAt   time.Time `json:"updatedAt"`

	Health          string    `json:"health"`
	HealthUpdatedAt time.Time `json:"healthUpdatedAt"`

	VersionInfo map[string]VersionInfo `json:"versionInfo"`
	Version     string                 `json:"version"`
	Revision    string                 // the GitCommit this maps to

	Scheduler bool `json:"scheduler"`

	Cordon bool `json:"cordon"`
	Drain  bool `json:"drain"`

	VolumeStats VolumeStats `json:"volumeStats"`

	// PoolStats     map[string]map[string]CapacityStats `json:"poolStats"`

	CapacityStats CapacityStats `json:"capacityStats"`
}

// NodeConfig is a read-only representation of the node's configuration, set at
// start time by environment variables passed to the container or using defaults.
type NodeConfig struct {
	// UUID is the unique identifier of the node.  It cannot be changed once set.
	ID string `json:"id,omitempty"`

	// Hostname of the node.
	Hostname string `json:"hostname"`

	// Address is is used for communication between nodes.
	// Nodes will fail to start if the address they first registered with
	// changes.  This protects against the container being re-scheduled on a
	// different host.  Nodes will typically use the host server's ip address,
	// running the docker container in -net host mode.
	Address string `json:"address"`

	// KvAddr is the address of the KV store to use for storing configuration.
	// It can include the address or FQDN with optional port.  Defaults to
	// Address/ADVERTISE_IP.
	KvAddr string `json:"kvAddr"`

	// Port allocations
	APIPort         int `json:"apiPort"`
	NatsPort        int `json:"natsPort"`
	NatsClusterPort int `json:"natsClusterPort"`
	SerfPort        int `json:"serfPort"`
	DFSPort         int `json:"dfsPort"`
	KVPeerPort      int `json:"kvPeerPort"`
	KVClientPort    int `json:"kvClientPort"`

	Labels map[string]string `json:"labels"`

	LogLevel  string `json:"logLevel"`  // the level of the logs to outout
	LogFormat string `json:"logFormat"` // either text or json
	LogFilter string `json:"logFilter"` // used to discard messages based on the message's category

	// BindAddr is used to control the default address StorageOS binds to.  This
	// should always be set to 0.0.0.0 (all interfaces).
	BindAddr string `json:"bindAddr"`

	// DeviceDir is where the volumes are exported.  This directory must be
	// shared into the container using the rshared volume mount option.
	DeviceDir string `json:"deviceDir"`

	// Join existing cluster
	Join string `json:"join"`

	// Backend selects the KV backend, either embedded (testing only) or etcd.
	Backend string `json:"kvBackend"`

	// EnableDebug is used to enable various debugging features.  Used by http
	// to enable debug endpoints and as a shortcut to enable debug logging.
	EnableDebug bool `json:"debug"`

	// Devices specify all devices that are available on the node.
	Devices []Device `json:"devices"`
}

// Device - device type
type Device struct {
	ID            string
	Labels        map[string]string `json:"labels"`
	Status        string            `json:"status"`
	Identifier    string            `json:"identifier"`
	Class         string            `json:"class"`
	CapacityStats CapacityStats     `json:"capacityStats"`
	CreatedAt     time.Time         `json:"createdAt"`
	UpdatedAt     time.Time         `json:"updatedAt"`
}
