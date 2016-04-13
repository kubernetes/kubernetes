package types

import (
	"os"
	"time"

	"github.com/docker/docker/daemon/network"
	"github.com/docker/docker/pkg/version"
	"github.com/docker/docker/runconfig"
)

// ContainerCreateResponse contains the information returned to a client on the
// creation of a new container.
type ContainerCreateResponse struct {
	// ID is the ID of the created container.
	ID string `json:"Id"`

	// Warnings are any warnings encountered during the creation of the container.
	Warnings []string `json:"Warnings"`
}

// POST /containers/{name:.*}/exec
type ContainerExecCreateResponse struct {
	// ID is the exec ID.
	ID string `json:"Id"`
}

// POST /auth
type AuthResponse struct {
	// Status is the authentication status
	Status string `json:"Status"`
}

// POST "/containers/"+containerID+"/wait"
type ContainerWaitResponse struct {
	// StatusCode is the status code of the wait job
	StatusCode int `json:"StatusCode"`
}

// POST "/commit?container="+containerID
type ContainerCommitResponse struct {
	ID string `json:"Id"`
}

// GET "/containers/{name:.*}/changes"
type ContainerChange struct {
	Kind int
	Path string
}

// GET "/images/{name:.*}/history"
type ImageHistory struct {
	ID        string `json:"Id"`
	Created   int64
	CreatedBy string
	Tags      []string
	Size      int64
	Comment   string
}

// DELETE "/images/{name:.*}"
type ImageDelete struct {
	Untagged string `json:",omitempty"`
	Deleted  string `json:",omitempty"`
}

// GET "/images/json"
type Image struct {
	ID          string `json:"Id"`
	ParentId    string
	RepoTags    []string
	RepoDigests []string
	Created     int
	Size        int
	VirtualSize int
	Labels      map[string]string
}

type GraphDriverData struct {
	Name string
	Data map[string]string
}

// GET "/images/{name:.*}/json"
type ImageInspect struct {
	Id              string
	Parent          string
	Comment         string
	Created         time.Time
	Container       string
	ContainerConfig *runconfig.Config
	DockerVersion   string
	Author          string
	Config          *runconfig.Config
	Architecture    string
	Os              string
	Size            int64
	VirtualSize     int64
	GraphDriver     GraphDriverData
}

// GET  "/containers/json"
type Port struct {
	IP          string `json:",omitempty"`
	PrivatePort int
	PublicPort  int `json:",omitempty"`
	Type        string
}

type Container struct {
	ID         string `json:"Id"`
	Names      []string
	Image      string
	Command    string
	Created    int
	Ports      []Port
	SizeRw     int `json:",omitempty"`
	SizeRootFs int `json:",omitempty"`
	Labels     map[string]string
	Status     string
	HostConfig struct {
		NetworkMode string `json:",omitempty"`
	}
}

// POST "/containers/"+containerID+"/copy"
type CopyConfig struct {
	Resource string
}

// ContainerPathStat is used to encode the header from
// 	GET /containers/{name:.*}/archive
// "name" is the file or directory name.
// "path" is the absolute path to the resource in the container.
type ContainerPathStat struct {
	Name  string      `json:"name"`
	Path  string      `json:"path"`
	Size  int64       `json:"size"`
	Mode  os.FileMode `json:"mode"`
	Mtime time.Time   `json:"mtime"`
}

// GET "/containers/{name:.*}/top"
type ContainerProcessList struct {
	Processes [][]string
	Titles    []string
}

type Version struct {
	Version       string
	ApiVersion    version.Version
	GitCommit     string
	GoVersion     string
	Os            string
	Arch          string
	KernelVersion string `json:",omitempty"`
	Experimental  bool   `json:",omitempty"`
	BuildTime     string `json:",omitempty"`
}

// GET "/info"
type Info struct {
	ID                 string
	Containers         int
	Images             int
	Driver             string
	DriverStatus       [][2]string
	MemoryLimit        bool
	SwapLimit          bool
	CpuCfsPeriod       bool
	CpuCfsQuota        bool
	IPv4Forwarding     bool
	BridgeNfIptables   bool
	BridgeNfIp6tables  bool
	Debug              bool
	NFd                int
	OomKillDisable     bool
	NGoroutines        int
	SystemTime         string
	ExecutionDriver    string
	LoggingDriver      string
	NEventsListener    int
	KernelVersion      string
	OperatingSystem    string
	IndexServerAddress string
	RegistryConfig     interface{}
	InitSha1           string
	InitPath           string
	NCPU               int
	MemTotal           int64
	DockerRootDir      string
	HttpProxy          string
	HttpsProxy         string
	NoProxy            string
	Name               string
	Labels             []string
	ExperimentalBuild  bool
}

// This struct is a temp struct used by execStart
// Config fields is part of ExecConfig in runconfig package
type ExecStartCheck struct {
	// ExecStart will first check if it's detached
	Detach bool
	// Check if there's a tty
	Tty bool
}

type ContainerState struct {
	Running    bool
	Paused     bool
	Restarting bool
	OOMKilled  bool
	Dead       bool
	Pid        int
	ExitCode   int
	Error      string
	StartedAt  time.Time
	FinishedAt time.Time
}

// GET "/containers/{name:.*}/json"
type ContainerJSONBase struct {
	Id              string
	Created         time.Time
	Path            string
	Args            []string
	State           *ContainerState
	Image           string
	NetworkSettings *network.Settings
	ResolvConfPath  string
	HostnamePath    string
	HostsPath       string
	LogPath         string
	Name            string
	RestartCount    int
	Driver          string
	ExecDriver      string
	MountLabel      string
	ProcessLabel    string
	AppArmorProfile string
	ExecIDs         []string
	HostConfig      *runconfig.HostConfig
	GraphDriver     GraphDriverData
}

type ContainerJSON struct {
	*ContainerJSONBase
	Mounts []MountPoint
	Config *runconfig.Config
}

// backcompatibility struct along with ContainerConfig
type ContainerJSONPre120 struct {
	*ContainerJSONBase
	Volumes   map[string]string
	VolumesRW map[string]bool
	Config    *ContainerConfig
}

type ContainerConfig struct {
	*runconfig.Config

	// backward compatibility, they now live in HostConfig
	Memory     int64
	MemorySwap int64
	CpuShares  int64
	Cpuset     string
}

// MountPoint represents a mount point configuration inside the container.
type MountPoint struct {
	Name        string `json:",omitempty"`
	Source      string
	Destination string
	Driver      string `json:",omitempty"`
	Mode        string // this is internally named `Relabel`
	RW          bool
}
