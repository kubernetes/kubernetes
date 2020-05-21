package docker10

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// DockerImage is the type representing a container image and its various properties when
// retrieved from the Docker client API.
type DockerImage struct {
	metav1.TypeMeta `json:",inline"`

	ID              string        `json:"Id"`
	Parent          string        `json:"Parent,omitempty"`
	Comment         string        `json:"Comment,omitempty"`
	Created         metav1.Time   `json:"Created,omitempty"`
	Container       string        `json:"Container,omitempty"`
	ContainerConfig DockerConfig  `json:"ContainerConfig,omitempty"`
	DockerVersion   string        `json:"DockerVersion,omitempty"`
	Author          string        `json:"Author,omitempty"`
	Config          *DockerConfig `json:"Config,omitempty"`
	Architecture    string        `json:"Architecture,omitempty"`
	Size            int64         `json:"Size,omitempty"`
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
	Labels          map[string]string   `json:"Labels,omitempty"`
}
