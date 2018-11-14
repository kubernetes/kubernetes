package types

import (
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/api/types/mount"
	"github.com/docker/docker/api/types/network"
	"github.com/docker/docker/api/types/registry"
	"github.com/docker/docker/api/types/swarm"
	"github.com/docker/go-connections/nat"
)

// RootFS returns Image's RootFS description including the layer IDs.
type RootFS struct {
	Type      string
	Layers    []string `json:",omitempty"`
	BaseLayer string   `json:",omitempty"`
}

// ImageInspect contains response of Engine API:
// GET "/images/{name:.*}/json"
type ImageInspect struct {
	ID              string `json:"Id"`
	RepoTags        []string
	RepoDigests     []string
	Parent          string
	Comment         string
	Created         string
	Container       string
	ContainerConfig *container.Config
	DockerVersion   string
	Author          string
	Config          *container.Config
	Architecture    string
	Os              string
	OsVersion       string `json:",omitempty"`
	Size            int64
	VirtualSize     int64
	GraphDriver     GraphDriverData
	RootFS          RootFS
	Metadata        ImageMetadata
}

// ImageMetadata contains engine-local data about the image
type ImageMetadata struct {
	LastTagTime time.Time `json:",omitempty"`
}

// Container contains response of Engine API:
// GET "/containers/json"
type Container struct {
	ID         string `json:"Id"`
	Names      []string
	Image      string
	ImageID    string
	Command    string
	Created    int64
	Ports      []Port
	SizeRw     int64 `json:",omitempty"`
	SizeRootFs int64 `json:",omitempty"`
	Labels     map[string]string
	State      string
	Status     string
	HostConfig struct {
		NetworkMode string `json:",omitempty"`
	}
	NetworkSettings *SummaryNetworkSettings
	Mounts          []MountPoint
}

// CopyConfig contains request body of Engine API:
// POST "/containers/"+containerID+"/copy"
type CopyConfig struct {
	Resource string
}

// ContainerPathStat is used to encode the header from
// GET "/containers/{name:.*}/archive"
// "Name" is the file or directory name.
type ContainerPathStat struct {
	Name       string      `json:"name"`
	Size       int64       `json:"size"`
	Mode       os.FileMode `json:"mode"`
	Mtime      time.Time   `json:"mtime"`
	LinkTarget string      `json:"linkTarget"`
}

// ContainerStats contains response of Engine API:
// GET "/stats"
type ContainerStats struct {
	Body   io.ReadCloser `json:"body"`
	OSType string        `json:"ostype"`
}

// Ping contains response of Engine API:
// GET "/_ping"
type Ping struct {
	APIVersion   string
	OSType       string
	Experimental bool
}

// ComponentVersion describes the version information for a specific component.
type ComponentVersion struct {
	Name    string
	Version string
	Details map[string]string `json:",omitempty"`
}

// Version contains response of Engine API:
// GET "/version"
type Version struct {
	Platform   struct{ Name string } `json:",omitempty"`
	Components []ComponentVersion    `json:",omitempty"`

	// The following fields are deprecated, they relate to the Engine component and are kept for backwards compatibility

	Version       string
	APIVersion    string `json:"ApiVersion"`
	MinAPIVersion string `json:"MinAPIVersion,omitempty"`
	GitCommit     string
	GoVersion     string
	Os            string
	Arch          string
	KernelVersion string `json:",omitempty"`
	Experimental  bool   `json:",omitempty"`
	BuildTime     string `json:",omitempty"`
}

// Commit holds the Git-commit (SHA1) that a binary was built from, as reported
// in the version-string of external tools, such as containerd, or runC.
type Commit struct {
	ID       string // ID is the actual commit ID of external tool.
	Expected string // Expected is the commit ID of external tool expected by dockerd as set at build time.
}

// Info contains response of Engine API:
// GET "/info"
type Info struct {
	ID                 string
	Containers         int
	ContainersRunning  int
	ContainersPaused   int
	ContainersStopped  int
	Images             int
	Driver             string
	DriverStatus       [][2]string
	SystemStatus       [][2]string
	Plugins            PluginsInfo
	MemoryLimit        bool
	SwapLimit          bool
	KernelMemory       bool
	CPUCfsPeriod       bool `json:"CpuCfsPeriod"`
	CPUCfsQuota        bool `json:"CpuCfsQuota"`
	CPUShares          bool
	CPUSet             bool
	IPv4Forwarding     bool
	BridgeNfIptables   bool
	BridgeNfIP6tables  bool `json:"BridgeNfIp6tables"`
	Debug              bool
	NFd                int
	OomKillDisable     bool
	NGoroutines        int
	SystemTime         string
	LoggingDriver      string
	CgroupDriver       string
	NEventsListener    int
	KernelVersion      string
	OperatingSystem    string
	OSType             string
	Architecture       string
	IndexServerAddress string
	RegistryConfig     *registry.ServiceConfig
	NCPU               int
	MemTotal           int64
	GenericResources   []swarm.GenericResource
	DockerRootDir      string
	HTTPProxy          string `json:"HttpProxy"`
	HTTPSProxy         string `json:"HttpsProxy"`
	NoProxy            string
	Name               string
	Labels             []string
	ExperimentalBuild  bool
	ServerVersion      string
	ClusterStore       string
	ClusterAdvertise   string
	Runtimes           map[string]Runtime
	DefaultRuntime     string
	Swarm              swarm.Info
	// LiveRestoreEnabled determines whether containers should be kept
	// running when the daemon is shutdown or upon daemon start if
	// running containers are detected
	LiveRestoreEnabled bool
	Isolation          container.Isolation
	InitBinary         string
	ContainerdCommit   Commit
	RuncCommit         Commit
	InitCommit         Commit
	SecurityOptions    []string
}

// KeyValue holds a key/value pair
type KeyValue struct {
	Key, Value string
}

// SecurityOpt contains the name and options of a security option
type SecurityOpt struct {
	Name    string
	Options []KeyValue
}

// DecodeSecurityOptions decodes a security options string slice to a type safe
// SecurityOpt
func DecodeSecurityOptions(opts []string) ([]SecurityOpt, error) {
	so := []SecurityOpt{}
	for _, opt := range opts {
		// support output from a < 1.13 docker daemon
		if !strings.Contains(opt, "=") {
			so = append(so, SecurityOpt{Name: opt})
			continue
		}
		secopt := SecurityOpt{}
		split := strings.Split(opt, ",")
		for _, s := range split {
			kv := strings.SplitN(s, "=", 2)
			if len(kv) != 2 {
				return nil, fmt.Errorf("invalid security option %q", s)
			}
			if kv[0] == "" || kv[1] == "" {
				return nil, errors.New("invalid empty security option")
			}
			if kv[0] == "name" {
				secopt.Name = kv[1]
				continue
			}
			secopt.Options = append(secopt.Options, KeyValue{Key: kv[0], Value: kv[1]})
		}
		so = append(so, secopt)
	}
	return so, nil
}

// PluginsInfo is a temp struct holding Plugins name
// registered with docker daemon. It is used by Info struct
type PluginsInfo struct {
	// List of Volume plugins registered
	Volume []string
	// List of Network plugins registered
	Network []string
	// List of Authorization plugins registered
	Authorization []string
	// List of Log plugins registered
	Log []string
}

// ExecStartCheck is a temp struct used by execStart
// Config fields is part of ExecConfig in runconfig package
type ExecStartCheck struct {
	// ExecStart will first check if it's detached
	Detach bool
	// Check if there's a tty
	Tty bool
}

// HealthcheckResult stores information about a single run of a healthcheck probe
type HealthcheckResult struct {
	Start    time.Time // Start is the time this check started
	End      time.Time // End is the time this check ended
	ExitCode int       // ExitCode meanings: 0=healthy, 1=unhealthy, 2=reserved (considered unhealthy), else=error running probe
	Output   string    // Output from last check
}

// Health states
const (
	NoHealthcheck = "none"      // Indicates there is no healthcheck
	Starting      = "starting"  // Starting indicates that the container is not yet ready
	Healthy       = "healthy"   // Healthy indicates that the container is running correctly
	Unhealthy     = "unhealthy" // Unhealthy indicates that the container has a problem
)

// Health stores information about the container's healthcheck results
type Health struct {
	Status        string               // Status is one of Starting, Healthy or Unhealthy
	FailingStreak int                  // FailingStreak is the number of consecutive failures
	Log           []*HealthcheckResult // Log contains the last few results (oldest first)
}

// ContainerState stores container's running state
// it's part of ContainerJSONBase and will return by "inspect" command
type ContainerState struct {
	Status     string // String representation of the container state. Can be one of "created", "running", "paused", "restarting", "removing", "exited", or "dead"
	Running    bool
	Paused     bool
	Restarting bool
	OOMKilled  bool
	Dead       bool
	Pid        int
	ExitCode   int
	Error      string
	StartedAt  string
	FinishedAt string
	Health     *Health `json:",omitempty"`
}

// ContainerNode stores information about the node that a container
// is running on.  It's only available in Docker Swarm
type ContainerNode struct {
	ID        string
	IPAddress string `json:"IP"`
	Addr      string
	Name      string
	Cpus      int
	Memory    int64
	Labels    map[string]string
}

// ContainerJSONBase contains response of Engine API:
// GET "/containers/{name:.*}/json"
type ContainerJSONBase struct {
	ID              string `json:"Id"`
	Created         string
	Path            string
	Args            []string
	State           *ContainerState
	Image           string
	ResolvConfPath  string
	HostnamePath    string
	HostsPath       string
	LogPath         string
	Node            *ContainerNode `json:",omitempty"`
	Name            string
	RestartCount    int
	Driver          string
	Platform        string
	MountLabel      string
	ProcessLabel    string
	AppArmorProfile string
	ExecIDs         []string
	HostConfig      *container.HostConfig
	GraphDriver     GraphDriverData
	SizeRw          *int64 `json:",omitempty"`
	SizeRootFs      *int64 `json:",omitempty"`
}

// ContainerJSON is newly used struct along with MountPoint
type ContainerJSON struct {
	*ContainerJSONBase
	Mounts          []MountPoint
	Config          *container.Config
	NetworkSettings *NetworkSettings
}

// NetworkSettings exposes the network settings in the api
type NetworkSettings struct {
	NetworkSettingsBase
	DefaultNetworkSettings
	Networks map[string]*network.EndpointSettings
}

// SummaryNetworkSettings provides a summary of container's networks
// in /containers/json
type SummaryNetworkSettings struct {
	Networks map[string]*network.EndpointSettings
}

// NetworkSettingsBase holds basic information about networks
type NetworkSettingsBase struct {
	Bridge                 string      // Bridge is the Bridge name the network uses(e.g. `docker0`)
	SandboxID              string      // SandboxID uniquely represents a container's network stack
	HairpinMode            bool        // HairpinMode specifies if hairpin NAT should be enabled on the virtual interface
	LinkLocalIPv6Address   string      // LinkLocalIPv6Address is an IPv6 unicast address using the link-local prefix
	LinkLocalIPv6PrefixLen int         // LinkLocalIPv6PrefixLen is the prefix length of an IPv6 unicast address
	Ports                  nat.PortMap // Ports is a collection of PortBinding indexed by Port
	SandboxKey             string      // SandboxKey identifies the sandbox
	SecondaryIPAddresses   []network.Address
	SecondaryIPv6Addresses []network.Address
}

// DefaultNetworkSettings holds network information
// during the 2 release deprecation period.
// It will be removed in Docker 1.11.
type DefaultNetworkSettings struct {
	EndpointID          string // EndpointID uniquely represents a service endpoint in a Sandbox
	Gateway             string // Gateway holds the gateway address for the network
	GlobalIPv6Address   string // GlobalIPv6Address holds network's global IPv6 address
	GlobalIPv6PrefixLen int    // GlobalIPv6PrefixLen represents mask length of network's global IPv6 address
	IPAddress           string // IPAddress holds the IPv4 address for the network
	IPPrefixLen         int    // IPPrefixLen represents mask length of network's IPv4 address
	IPv6Gateway         string // IPv6Gateway holds gateway address specific for IPv6
	MacAddress          string // MacAddress holds the MAC address for the network
}

// MountPoint represents a mount point configuration inside the container.
// This is used for reporting the mountpoints in use by a container.
type MountPoint struct {
	Type        mount.Type `json:",omitempty"`
	Name        string     `json:",omitempty"`
	Source      string
	Destination string
	Driver      string `json:",omitempty"`
	Mode        string
	RW          bool
	Propagation mount.Propagation
}

// NetworkResource is the body of the "get network" http response message
type NetworkResource struct {
	Name       string                         // Name is the requested name of the network
	ID         string                         `json:"Id"` // ID uniquely identifies a network on a single machine
	Created    time.Time                      // Created is the time the network created
	Scope      string                         // Scope describes the level at which the network exists (e.g. `swarm` for cluster-wide or `local` for machine level)
	Driver     string                         // Driver is the Driver name used to create the network (e.g. `bridge`, `overlay`)
	EnableIPv6 bool                           // EnableIPv6 represents whether to enable IPv6
	IPAM       network.IPAM                   // IPAM is the network's IP Address Management
	Internal   bool                           // Internal represents if the network is used internal only
	Attachable bool                           // Attachable represents if the global scope is manually attachable by regular containers from workers in swarm mode.
	Ingress    bool                           // Ingress indicates the network is providing the routing-mesh for the swarm cluster.
	ConfigFrom network.ConfigReference        // ConfigFrom specifies the source which will provide the configuration for this network.
	ConfigOnly bool                           // ConfigOnly networks are place-holder networks for network configurations to be used by other networks. ConfigOnly networks cannot be used directly to run containers or services.
	Containers map[string]EndpointResource    // Containers contains endpoints belonging to the network
	Options    map[string]string              // Options holds the network specific options to use for when creating the network
	Labels     map[string]string              // Labels holds metadata specific to the network being created
	Peers      []network.PeerInfo             `json:",omitempty"` // List of peer nodes for an overlay network
	Services   map[string]network.ServiceInfo `json:",omitempty"`
}

// EndpointResource contains network resources allocated and used for a container in a network
type EndpointResource struct {
	Name        string
	EndpointID  string
	MacAddress  string
	IPv4Address string
	IPv6Address string
}

// NetworkCreate is the expected body of the "create network" http request message
type NetworkCreate struct {
	// Check for networks with duplicate names.
	// Network is primarily keyed based on a random ID and not on the name.
	// Network name is strictly a user-friendly alias to the network
	// which is uniquely identified using ID.
	// And there is no guaranteed way to check for duplicates.
	// Option CheckDuplicate is there to provide a best effort checking of any networks
	// which has the same name but it is not guaranteed to catch all name collisions.
	CheckDuplicate bool
	Driver         string
	Scope          string
	EnableIPv6     bool
	IPAM           *network.IPAM
	Internal       bool
	Attachable     bool
	Ingress        bool
	ConfigOnly     bool
	ConfigFrom     *network.ConfigReference
	Options        map[string]string
	Labels         map[string]string
}

// NetworkCreateRequest is the request message sent to the server for network create call.
type NetworkCreateRequest struct {
	NetworkCreate
	Name string
}

// NetworkCreateResponse is the response message sent by the server for network create call
type NetworkCreateResponse struct {
	ID      string `json:"Id"`
	Warning string
}

// NetworkConnect represents the data to be used to connect a container to the network
type NetworkConnect struct {
	Container      string
	EndpointConfig *network.EndpointSettings `json:",omitempty"`
}

// NetworkDisconnect represents the data to be used to disconnect a container from the network
type NetworkDisconnect struct {
	Container string
	Force     bool
}

// NetworkInspectOptions holds parameters to inspect network
type NetworkInspectOptions struct {
	Scope   string
	Verbose bool
}

// Checkpoint represents the details of a checkpoint
type Checkpoint struct {
	Name string // Name is the name of the checkpoint
}

// Runtime describes an OCI runtime
type Runtime struct {
	Path string   `json:"path"`
	Args []string `json:"runtimeArgs,omitempty"`
}

// DiskUsage contains response of Engine API:
// GET "/system/df"
type DiskUsage struct {
	LayersSize  int64
	Images      []*ImageSummary
	Containers  []*Container
	Volumes     []*Volume
	BuilderSize int64
}

// ContainersPruneReport contains the response for Engine API:
// POST "/containers/prune"
type ContainersPruneReport struct {
	ContainersDeleted []string
	SpaceReclaimed    uint64
}

// VolumesPruneReport contains the response for Engine API:
// POST "/volumes/prune"
type VolumesPruneReport struct {
	VolumesDeleted []string
	SpaceReclaimed uint64
}

// ImagesPruneReport contains the response for Engine API:
// POST "/images/prune"
type ImagesPruneReport struct {
	ImagesDeleted  []ImageDeleteResponseItem
	SpaceReclaimed uint64
}

// BuildCachePruneReport contains the response for Engine API:
// POST "/build/prune"
type BuildCachePruneReport struct {
	SpaceReclaimed uint64
}

// NetworksPruneReport contains the response for Engine API:
// POST "/networks/prune"
type NetworksPruneReport struct {
	NetworksDeleted []string
}

// SecretCreateResponse contains the information returned to a client
// on the creation of a new secret.
type SecretCreateResponse struct {
	// ID is the id of the created secret.
	ID string
}

// SecretListOptions holds parameters to list secrets
type SecretListOptions struct {
	Filters filters.Args
}

// ConfigCreateResponse contains the information returned to a client
// on the creation of a new config.
type ConfigCreateResponse struct {
	// ID is the id of the created config.
	ID string
}

// ConfigListOptions holds parameters to list configs
type ConfigListOptions struct {
	Filters filters.Args
}

// PushResult contains the tag, manifest digest, and manifest size from the
// push. It's used to signal this information to the trust code in the client
// so it can sign the manifest if necessary.
type PushResult struct {
	Tag    string
	Digest string
	Size   int
}

// BuildResult contains the image id of a successful build
type BuildResult struct {
	ID string
}
