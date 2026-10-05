package specs

import "os"

// Spec is the base configuration for CDI
type Spec struct {
	Version string `json:"cdiVersion" yaml:"cdiVersion"`
	Kind    string `json:"kind"       yaml:"kind"`
	// Annotations add meta information per CDI spec. Note these are CDI-specific and do not affect container metadata.
	// Added in v0.6.0.
	Annotations    map[string]string `json:"annotations,omitempty"    yaml:"annotations,omitempty"`
	Devices        []Device          `json:"devices"                  yaml:"devices"`
	ContainerEdits ContainerEdits    `json:"containerEdits,omitempty" yaml:"containerEdits,omitempty"`
}

// Device is a "Device" a container runtime can add to a container
type Device struct {
	Name string `json:"name" yaml:"name"`
	// Annotations add meta information per device. Note these are CDI-specific and do not affect container metadata.
	// Added in v0.6.0.
	Annotations    map[string]string `json:"annotations,omitempty" yaml:"annotations,omitempty"`
	ContainerEdits ContainerEdits    `json:"containerEdits"        yaml:"containerEdits"`
}

// ContainerEdits are edits a container runtime must make to the OCI spec to expose the device.
type ContainerEdits struct {
	Env            []string          `json:"env,omitempty"            yaml:"env,omitempty"`
	DeviceNodes    []*DeviceNode     `json:"deviceNodes,omitempty"    yaml:"deviceNodes,omitempty"`
	NetDevices     []*LinuxNetDevice `json:"netDevices,omitempty"     yaml:"netDevices,omitempty"` // Added in v1.1.0
	Hooks          []*Hook           `json:"hooks,omitempty"          yaml:"hooks,omitempty"`
	Mounts         []*Mount          `json:"mounts,omitempty"         yaml:"mounts,omitempty"`
	IntelRdt       *IntelRdt         `json:"intelRdt,omitempty"       yaml:"intelRdt,omitempty"`       // Added in v0.7.0
	AdditionalGIDs []uint32          `json:"additionalGids,omitempty" yaml:"additionalGids,omitempty"` // Added in v0.7.0
}

// DeviceNode represents a device node that needs to be added to the OCI spec.
type DeviceNode struct {
	Path        string       `json:"path"                  yaml:"path"`
	HostPath    string       `json:"hostPath,omitempty"    yaml:"hostPath,omitempty"` // Added in v0.5.0
	Type        string       `json:"type,omitempty"        yaml:"type,omitempty"`
	Major       int64        `json:"major,omitempty"       yaml:"major,omitempty"`
	Minor       int64        `json:"minor,omitempty"       yaml:"minor,omitempty"`
	FileMode    *os.FileMode `json:"fileMode,omitempty"    yaml:"fileMode,omitempty"`
	Permissions string       `json:"permissions,omitempty" yaml:"permissions,omitempty"`
	UID         *uint32      `json:"uid,omitempty"         yaml:"uid,omitempty"`
	GID         *uint32      `json:"gid,omitempty"         yaml:"gid,omitempty"`
}

// Mount represents a mount that needs to be added to the OCI spec.
type Mount struct {
	HostPath      string   `json:"hostPath"          yaml:"hostPath"`
	ContainerPath string   `json:"containerPath"     yaml:"containerPath"`
	Options       []string `json:"options,omitempty" yaml:"options,omitempty"`
	Type          string   `json:"type,omitempty"    yaml:"type,omitempty"` // Added in v0.4.0
}

// Hook represents a hook that needs to be added to the OCI spec.
type Hook struct {
	HookName string   `json:"hookName"          yaml:"hookName"`
	Path     string   `json:"path"              yaml:"path"`
	Args     []string `json:"args,omitempty"    yaml:"args,omitempty"`
	Env      []string `json:"env,omitempty"     yaml:"env,omitempty"`
	Timeout  *int     `json:"timeout,omitempty" yaml:"timeout,omitempty"`
}

// IntelRdt describes the Linux IntelRdt parameters to set in the OCI spec.
type IntelRdt struct {
	ClosID           string   `json:"closID,omitempty"           yaml:"closID,omitempty"`
	L3CacheSchema    string   `json:"l3CacheSchema,omitempty"    yaml:"l3CacheSchema,omitempty"`
	MemBwSchema      string   `json:"memBwSchema,omitempty"      yaml:"memBwSchema,omitempty"`
	Schemata         []string `json:"schemata,omitempty"         yaml:"schemata,omitempty"`         // Added in v1.1.0.
	EnableMonitoring bool     `json:"enableMonitoring,omitempty" yaml:"enableMonitoring,omitempty"` // Added in v1.1.0.
}

// LinuxNetDevice represents an OCI LinuxNetDevice to be added to the OCI Spec.
type LinuxNetDevice struct {
	HostInterfaceName string `json:"hostInterfaceName" yaml:"hostInterfaceName"`
	Name              string `json:"name"   yaml:"name"`
}
