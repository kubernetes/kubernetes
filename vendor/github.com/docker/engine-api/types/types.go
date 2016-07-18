package types

import (
	"os"
	"time"

	"github.com/docker/engine-api/types/container"
	"github.com/docker/engine-api/types/network"
	"github.com/docker/engine-api/types/registry"
	"github.com/docker/engine-api/types/swarm"
	"github.com/docker/go-connections/nat"
)

// ContainerCreateResponse contains the information returned to a client on the
// creation of a new container.
type ContainerCreateResponse struct {
	// ID is the ID of the created container.
	ID string `json:"Id"`

	// Warnings are any warnings encountered during the creation of the container.
	Warnings []string `json:"Warnings"`
}

// ContainerExecCreateResponse contains response of Remote API:
// POST "/containers/{name:.*}/exec"
type ContainerExecCreateResponse struct {
	// ID is the exec ID.
	ID string `json:"Id"`
}

// ContainerUpdateResponse contains response of Remote API:
// POST "/containers/{name:.*}/update"
type ContainerUpdateResponse struct {
	// Warnings are any warnings encountered during the updating of the container.
	Warnings []string `json:"Warnings"`
}

// AuthResponse contains response of Remote API:
// POST "/auth"
type AuthResponse struct {
	// Status is the authentication status
	Status string `json:"Status"`

	// IdentityToken is an opaque token used for authenticating
	// a user after a successful login.
	IdentityToken string `json:"IdentityToken,omitempty"`
}

// ContainerWaitResponse contains response of Remote API:
// POST "/containers/"+containerID+"/wait"
type ContainerWaitResponse struct {
	// StatusCode is the status code of the wait job
	StatusCode int `json:"StatusCode"`
}

// ContainerCommitResponse contains response of Remote API:
// POST "/commit?container="+containerID
type ContainerCommitResponse struct {
	ID string `json:"Id"`
}

// ContainerChange contains response of Remote API:
// GET "/containers/{name:.*}/changes"
type ContainerChange struct {
	Kind int
	Path string
}

// ImageHistory contains response of Remote API:
// GET "/images/{name:.*}/history"
type ImageHistory struct {
	ID        string `json:"Id"`
	Created   int64
	CreatedBy string
	Tags      []string
	Size      int64
	Comment   string
}

// ImageDelete contains response of Remote API:
// DELETE "/images/{name:.*}"
type ImageDelete struct {
	Untagged string `json:",omitempty"`
	Deleted  string `json:",omitempty"`
}

// Image contains response of Remote API:
// GET "/images/json"
type Image struct {
	ID          string `json:"Id"`
	ParentID    string `json:"ParentId"`
	RepoTags    []string
	RepoDigests []string
	Created     int64
	Size        int64
	VirtualSize int64
	Labels      map[string]string
}

// GraphDriverData returns Image's graph driver config info
// when calling inspect command
type GraphDriverData struct {
	Name string
	Data map[string]string
}

// RootFS returns Image's RootFS description including the layer IDs.
type RootFS struct {
	Type      string
	Layers    []string `json:",omitempty"`
	BaseLayer string   `json:",omitempty"`
}

// ImageInspect contains response of Remote API:
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
	Size            int64
	VirtualSize     int64
	GraphDriver     GraphDriverData
	RootFS          RootFS
}

// Port stores open ports info of container
// e.g. {"PrivatePort": 8080, "PublicPort": 80, "Type": "tcp"}
type Port struct {
	IP          string `json:",omitempty"`
	PrivatePort int
	PublicPort  int `json:",omitempty"`
	Type        string
}

// Container contains response of Remote API:
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

// CopyConfig contains request body of Remote API:
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

// ContainerProcessList contains response of Remote API:
// GET "/containers/{name:.*}/top"
type ContainerProcessList struct {
	Processes [][]string
	Titles    []string
}

// Version contains response of Remote API:
// GET "/version"
type Version struct {
	Version       string
	APIVersion    string `json:"ApiVersion"`
	GitCommit     string
	GoVersion     string
	Os            string
	Arch          string
	KernelVersion string `json:",omitempty"`
	Experimental  bool   `json:",omitempty"`
	BuildTime     string `json:",omitempty"`
}

// Info contains response of Remote API:
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
	ExecutionDriver    string
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
	SecurityOptions    []string
	Runtimes           map[string]Runtime
	DefaultRuntime     string
	Swarm              swarm.Info
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
	ExitCode int       // ExitCode meanings: 0=healthy, 1=unhealthy, 2=starting, else=error running probe
	Output   string    // Output from last check
}

// Health states
const (
	Starting  = "starting"  // Starting indicates that the container is not yet ready
	Healthy   = "healthy"   // Healthy indicates that the container is running correctly
	Unhealthy = "unhealthy" // Unhealthy indicates that the container has a problem
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
	Status     string
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

// ContainerJSONBase contains response of Remote API:
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
type MountPoint struct {
	Name        string `json:",omitempty"`
	Source      string
	Destination string
	Driver      string `json:",omitempty"`
	Mode        string
	RW          bool
	Propagation string
}

// Volume represents the configuration of a volume for the remote API
type Volume struct {
	Name       string                 // Name is the name of the volume
	Driver     string                 // Driver is the Driver name used to create the volume
	Mountpoint string                 // Mountpoint is the location on disk of the volume
	Status     map[string]interface{} `json:",omitempty"` // Status provides low-level status information about the volume
	Labels     map[string]string      // Labels is metadata specific to the volume
	Scope      string                 // Scope describes the level at which the volume exists (e.g. `global` for cluster-wide or `local` for machine level)
}

// VolumesListResponse contains the response for the remote API:
// GET "/volumes"
type VolumesListResponse struct {
	Volumes  []*Volume // Volumes is the list of volumes being returned
	Warnings []string  // Warnings is a list of warnings that occurred when getting the list from the volume drivers
}

// VolumeCreateRequest contains the response for the remote API:
// POST "/volumes/create"
type VolumeCreateRequest struct {
	Name       string            // Name is the requested name of the volume
	Driver     string            // Driver is the name of the driver that should be used to create the volume
	DriverOpts map[string]string // DriverOpts holds the driver specific options to use for when creating the volume.
	Labels     map[string]string // Labels holds metadata specific to the volume being created.
}

// NetworkResource is the body of the "get network" http response message
type NetworkResource struct {
	Name       string                      // Name is the requested name of the network
	ID         string                      `json:"Id"` // ID uniquely identifies a network on a single machine
	Scope      string                      // Scope describes the level at which the network exists (e.g. `global` for cluster-wide or `local` for machine level)
	Driver     string                      // Driver is the Driver name used to create the network (e.g. `bridge`, `overlay`)
	EnableIPv6 bool                        // EnableIPv6 represents whether to enable IPv6
	IPAM       network.IPAM                // IPAM is the network's IP Address Management
	Internal   bool                        // Internal represents if the network is used internal only
	Containers map[string]EndpointResource // Containers contains endpoints belonging to the network
	Options    map[string]string           // Options holds the network specific options to use for when creating the network
	Labels     map[string]string           // Labels holds metadata specific to the network being created
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
	CheckDuplicate bool
	Driver         string
	EnableIPv6     bool
	IPAM           network.IPAM
	Internal       bool
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

// Checkpoint represents the details of a checkpoint
type Checkpoint struct {
	Name string // Name is the name of the checkpoint
}

// Runtime describes an OCI runtime
type Runtime struct {
	Path string   `json:"path"`
	Args []string `json:"runtimeArgs,omitempty"`
}
