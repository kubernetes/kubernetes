package runconfig

import (
	"encoding/json"
	"io"
	"strings"

	"github.com/docker/docker/pkg/nat"
	"github.com/docker/docker/pkg/ulimit"
)

type KeyValuePair struct {
	Key   string
	Value string
}

type NetworkMode string

type IpcMode string

// IsPrivate indicates whether container use it's private ipc stack
func (n IpcMode) IsPrivate() bool {
	return !(n.IsHost() || n.IsContainer())
}

func (n IpcMode) IsHost() bool {
	return n == "host"
}

func (n IpcMode) IsContainer() bool {
	parts := strings.SplitN(string(n), ":", 2)
	return len(parts) > 1 && parts[0] == "container"
}

func (n IpcMode) Valid() bool {
	parts := strings.Split(string(n), ":")
	switch mode := parts[0]; mode {
	case "", "host":
	case "container":
		if len(parts) != 2 || parts[1] == "" {
			return false
		}
	default:
		return false
	}
	return true
}

func (n IpcMode) Container() string {
	parts := strings.SplitN(string(n), ":", 2)
	if len(parts) > 1 {
		return parts[1]
	}
	return ""
}

type UTSMode string

// IsPrivate indicates whether container use it's private UTS namespace
func (n UTSMode) IsPrivate() bool {
	return !(n.IsHost())
}

func (n UTSMode) IsHost() bool {
	return n == "host"
}

func (n UTSMode) Valid() bool {
	parts := strings.Split(string(n), ":")
	switch mode := parts[0]; mode {
	case "", "host":
	default:
		return false
	}
	return true
}

type PidMode string

// IsPrivate indicates whether container use it's private pid stack
func (n PidMode) IsPrivate() bool {
	return !(n.IsHost())
}

func (n PidMode) IsHost() bool {
	return n == "host"
}

func (n PidMode) Valid() bool {
	parts := strings.Split(string(n), ":")
	switch mode := parts[0]; mode {
	case "", "host":
	default:
		return false
	}
	return true
}

type DeviceMapping struct {
	PathOnHost        string
	PathInContainer   string
	CgroupPermissions string
}

type RestartPolicy struct {
	Name              string
	MaximumRetryCount int
}

func (rp *RestartPolicy) IsNone() bool {
	return rp.Name == "no"
}

func (rp *RestartPolicy) IsAlways() bool {
	return rp.Name == "always"
}

func (rp *RestartPolicy) IsOnFailure() bool {
	return rp.Name == "on-failure"
}

type LogConfig struct {
	Type   string
	Config map[string]string
}

type LxcConfig struct {
	values []KeyValuePair
}

func (c *LxcConfig) MarshalJSON() ([]byte, error) {
	if c == nil {
		return []byte{}, nil
	}
	return json.Marshal(c.Slice())
}

func (c *LxcConfig) UnmarshalJSON(b []byte) error {
	if len(b) == 0 {
		return nil
	}

	var kv []KeyValuePair
	if err := json.Unmarshal(b, &kv); err != nil {
		var h map[string]string
		if err := json.Unmarshal(b, &h); err != nil {
			return err
		}
		for k, v := range h {
			kv = append(kv, KeyValuePair{k, v})
		}
	}
	c.values = kv

	return nil
}

func (c *LxcConfig) Len() int {
	if c == nil {
		return 0
	}
	return len(c.values)
}

func (c *LxcConfig) Slice() []KeyValuePair {
	if c == nil {
		return nil
	}
	return c.values
}

func NewLxcConfig(values []KeyValuePair) *LxcConfig {
	return &LxcConfig{values}
}

type CapList struct {
	caps []string
}

func (c *CapList) MarshalJSON() ([]byte, error) {
	if c == nil {
		return []byte{}, nil
	}
	return json.Marshal(c.Slice())
}

func (c *CapList) UnmarshalJSON(b []byte) error {
	if len(b) == 0 {
		return nil
	}

	var caps []string
	if err := json.Unmarshal(b, &caps); err != nil {
		var s string
		if err := json.Unmarshal(b, &s); err != nil {
			return err
		}
		caps = append(caps, s)
	}
	c.caps = caps

	return nil
}

func (c *CapList) Len() int {
	if c == nil {
		return 0
	}
	return len(c.caps)
}

func (c *CapList) Slice() []string {
	if c == nil {
		return nil
	}
	return c.caps
}

func NewCapList(caps []string) *CapList {
	return &CapList{caps}
}

type HostConfig struct {
	Binds            []string
	ContainerIDFile  string
	LxcConf          *LxcConfig
	Memory           int64 // Memory limit (in bytes)
	MemorySwap       int64 // Total memory usage (memory + swap); set `-1` to disable swap
	CpuShares        int64 // CPU shares (relative weight vs. other containers)
	CpuPeriod        int64
	CpusetCpus       string // CpusetCpus 0-2, 0,1
	CpusetMems       string // CpusetMems 0-2, 0,1
	CpuQuota         int64
	BlkioWeight      int64 // Block IO weight (relative weight vs. other containers)
	OomKillDisable   bool  // Whether to disable OOM Killer or not
	MemorySwappiness int64 // Tuning container memory swappiness behaviour
	Privileged       bool
	PortBindings     nat.PortMap
	Links            []string
	PublishAllPorts  bool
	Dns              []string
	DnsSearch        []string
	ExtraHosts       []string
	VolumesFrom      []string
	Devices          []DeviceMapping
	NetworkMode      NetworkMode
	IpcMode          IpcMode
	PidMode          PidMode
	UTSMode          UTSMode
	CapAdd           *CapList
	CapDrop          *CapList
	GroupAdd         []string
	RestartPolicy    RestartPolicy
	SecurityOpt      []string
	ReadonlyRootfs   bool
	Ulimits          []*ulimit.Ulimit
	LogConfig        LogConfig
	CgroupParent     string // Parent cgroup.
	ConsoleSize      [2]int // Initial console size on Windows
}

func MergeConfigs(config *Config, hostConfig *HostConfig) *ContainerConfigWrapper {
	return &ContainerConfigWrapper{
		config,
		hostConfig,
		"", nil,
	}
}

func DecodeHostConfig(src io.Reader) (*HostConfig, error) {
	decoder := json.NewDecoder(src)

	var w ContainerConfigWrapper
	if err := decoder.Decode(&w); err != nil {
		return nil, err
	}

	hc := w.GetHostConfig()

	return hc, nil
}
