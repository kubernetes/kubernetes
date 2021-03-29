package dockerpre012

import (
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DockerImage is for earlier versions of the Docker API (pre-012 to be specific). It is also the
// version of metadata that the container image registry uses to persist metadata.
type DockerImage struct {
	metav1.TypeMeta `json:",inline"`

	ID              string        `json:"id"`
	Parent          string        `json:"parent,omitempty"`
	Comment         string        `json:"comment,omitempty"`
	Created         metav1.Time   `json:"created"`
	Container       string        `json:"container,omitempty"`
	ContainerConfig DockerConfig  `json:"container_config,omitempty"`
	DockerVersion   string        `json:"docker_version,omitempty"`
	Author          string        `json:"author,omitempty"`
	Config          *DockerConfig `json:"config,omitempty"`
	Architecture    string        `json:"architecture,omitempty"`
	Size            int64         `json:"size,omitempty"`
}

// DockerConfig is the list of configuration options used when creating a container.
type DockerConfig struct {
	Hostname        string              `json:"Hostname,omitempty"`
	Domainname      string              `json:"Domainname,omitempty"`
	User            string              `json:"User,omitempty"`
	Memory          int64               `json:"Memory,omitempty"`
	MemorySwap      int64               `json:"MemorySwap,omitempty"`
	CPUShares       int64               `json:"CpuShares,omitempty"`
	CPUSet          string              `json:"Cpuset,omitempty"`
	AttachStdin     bool                `json:"AttachStdin,omitempty"`
	AttachStdout    bool                `json:"AttachStdout,omitempty"`
	AttachStderr    bool                `json:"AttachStderr,omitempty"`
	PortSpecs       []string            `json:"PortSpecs,omitempty"`
	ExposedPorts    map[string]struct{} `json:"ExposedPorts,omitempty"`
	Tty             bool                `json:"Tty,omitempty"`
	OpenStdin       bool                `json:"OpenStdin,omitempty"`
	StdinOnce       bool                `json:"StdinOnce,omitempty"`
	Env             []string            `json:"Env,omitempty"`
	Cmd             []string            `json:"Cmd,omitempty"`
	DNS             []string            `json:"Dns,omitempty"` // For Docker API v1.9 and below only
	Image           string              `json:"Image,omitempty"`
	Volumes         map[string]struct{} `json:"Volumes,omitempty"`
	VolumesFrom     string              `json:"VolumesFrom,omitempty"`
	WorkingDir      string              `json:"WorkingDir,omitempty"`
	Entrypoint      []string            `json:"Entrypoint,omitempty"`
	NetworkDisabled bool                `json:"NetworkDisabled,omitempty"`
	SecurityOpts    []string            `json:"SecurityOpts,omitempty"`
	OnBuild         []string            `json:"OnBuild,omitempty"`
	// This field is not supported in pre012 and will always be empty.
	Labels map[string]string `json:"Labels,omitempty"`
}

// ImagePre012 serves the same purpose as the Image type except that it is for
// earlier versions of the Docker API (pre-012 to be specific)
// Exists only for legacy conversion, copy of type from fsouza/go-dockerclient
type ImagePre012 struct {
	ID              string    `json:"id"`
	Parent          string    `json:"parent,omitempty"`
	Comment         string    `json:"comment,omitempty"`
	Created         time.Time `json:"created"`
	Container       string    `json:"container,omitempty"`
	ContainerConfig Config    `json:"container_config,omitempty"`
	DockerVersion   string    `json:"docker_version,omitempty"`
	Author          string    `json:"author,omitempty"`
	Config          *Config   `json:"config,omitempty"`
	Architecture    string    `json:"architecture,omitempty"`
	Size            int64     `json:"size,omitempty"`
}

// Config is the list of configuration options used when creating a container.
// Config does not contain the options that are specific to starting a container on a
// given host.  Those are contained in HostConfig
// Exists only for legacy conversion, copy of type from fsouza/go-dockerclient
type Config struct {
	Hostname          string              `json:"Hostname,omitempty" yaml:"Hostname,omitempty"`
	Domainname        string              `json:"Domainname,omitempty" yaml:"Domainname,omitempty"`
	User              string              `json:"User,omitempty" yaml:"User,omitempty"`
	Memory            int64               `json:"Memory,omitempty" yaml:"Memory,omitempty"`
	MemorySwap        int64               `json:"MemorySwap,omitempty" yaml:"MemorySwap,omitempty"`
	MemoryReservation int64               `json:"MemoryReservation,omitempty" yaml:"MemoryReservation,omitempty"`
	KernelMemory      int64               `json:"KernelMemory,omitempty" yaml:"KernelMemory,omitempty"`
	PidsLimit         int64               `json:"PidsLimit,omitempty" yaml:"PidsLimit,omitempty"`
	CPUShares         int64               `json:"CpuShares,omitempty" yaml:"CpuShares,omitempty"`
	CPUSet            string              `json:"Cpuset,omitempty" yaml:"Cpuset,omitempty"`
	AttachStdin       bool                `json:"AttachStdin,omitempty" yaml:"AttachStdin,omitempty"`
	AttachStdout      bool                `json:"AttachStdout,omitempty" yaml:"AttachStdout,omitempty"`
	AttachStderr      bool                `json:"AttachStderr,omitempty" yaml:"AttachStderr,omitempty"`
	PortSpecs         []string            `json:"PortSpecs,omitempty" yaml:"PortSpecs,omitempty"`
	ExposedPorts      map[Port]struct{}   `json:"ExposedPorts,omitempty" yaml:"ExposedPorts,omitempty"`
	StopSignal        string              `json:"StopSignal,omitempty" yaml:"StopSignal,omitempty"`
	Tty               bool                `json:"Tty,omitempty" yaml:"Tty,omitempty"`
	OpenStdin         bool                `json:"OpenStdin,omitempty" yaml:"OpenStdin,omitempty"`
	StdinOnce         bool                `json:"StdinOnce,omitempty" yaml:"StdinOnce,omitempty"`
	Env               []string            `json:"Env,omitempty" yaml:"Env,omitempty"`
	Cmd               []string            `json:"Cmd" yaml:"Cmd"`
	DNS               []string            `json:"Dns,omitempty" yaml:"Dns,omitempty"` // For Docker API v1.9 and below only
	Image             string              `json:"Image,omitempty" yaml:"Image,omitempty"`
	Volumes           map[string]struct{} `json:"Volumes,omitempty" yaml:"Volumes,omitempty"`
	VolumeDriver      string              `json:"VolumeDriver,omitempty" yaml:"VolumeDriver,omitempty"`
	VolumesFrom       string              `json:"VolumesFrom,omitempty" yaml:"VolumesFrom,omitempty"`
	WorkingDir        string              `json:"WorkingDir,omitempty" yaml:"WorkingDir,omitempty"`
	MacAddress        string              `json:"MacAddress,omitempty" yaml:"MacAddress,omitempty"`
	Entrypoint        []string            `json:"Entrypoint" yaml:"Entrypoint"`
	NetworkDisabled   bool                `json:"NetworkDisabled,omitempty" yaml:"NetworkDisabled,omitempty"`
	SecurityOpts      []string            `json:"SecurityOpts,omitempty" yaml:"SecurityOpts,omitempty"`
	OnBuild           []string            `json:"OnBuild,omitempty" yaml:"OnBuild,omitempty"`
	Mounts            []Mount             `json:"Mounts,omitempty" yaml:"Mounts,omitempty"`
	Labels            map[string]string   `json:"Labels,omitempty" yaml:"Labels,omitempty"`
}

// Mount represents a mount point in the container.
//
// It has been added in the version 1.20 of the Docker API, available since
// Docker 1.8.
// Exists only for legacy conversion, copy of type from fsouza/go-dockerclient
type Mount struct {
	Name        string
	Source      string
	Destination string
	Driver      string
	Mode        string
	RW          bool
}

// Port represents the port number and the protocol, in the form
// <number>/<protocol>. For example: 80/tcp.
// Exists only for legacy conversion, copy of type from fsouza/go-dockerclient
type Port string
