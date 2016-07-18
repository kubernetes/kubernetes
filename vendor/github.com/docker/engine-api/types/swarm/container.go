package swarm

import "time"

// ContainerSpec represents the spec of a container.
type ContainerSpec struct {
	Image           string            `json:",omitempty"`
	Labels          map[string]string `json:",omitempty"`
	Command         []string          `json:",omitempty"`
	Args            []string          `json:",omitempty"`
	Env             []string          `json:",omitempty"`
	Dir             string            `json:",omitempty"`
	User            string            `json:",omitempty"`
	Mounts          []Mount           `json:",omitempty"`
	StopGracePeriod *time.Duration    `json:",omitempty"`
}

// MountType represents the type of a mount.
type MountType string

const (
	// MountTypeBind BIND
	MountTypeBind MountType = "bind"
	// MountTypeVolume VOLUME
	MountTypeVolume MountType = "volume"
)

// Mount represents a mount (volume).
type Mount struct {
	Type     MountType `json:",omitempty"`
	Source   string    `json:",omitempty"`
	Target   string    `json:",omitempty"`
	ReadOnly bool      `json:",omitempty"`

	BindOptions   *BindOptions   `json:",omitempty"`
	VolumeOptions *VolumeOptions `json:",omitempty"`
}

// MountPropagation represents the propagation of a mount.
type MountPropagation string

const (
	// MountPropagationRPrivate RPRIVATE
	MountPropagationRPrivate MountPropagation = "rprivate"
	// MountPropagationPrivate PRIVATE
	MountPropagationPrivate MountPropagation = "private"
	// MountPropagationRShared RSHARED
	MountPropagationRShared MountPropagation = "rshared"
	// MountPropagationShared SHARED
	MountPropagationShared MountPropagation = "shared"
	// MountPropagationRSlave RSLAVE
	MountPropagationRSlave MountPropagation = "rslave"
	// MountPropagationSlave SLAVE
	MountPropagationSlave MountPropagation = "slave"
)

// BindOptions defines options specific to mounts of type "bind".
type BindOptions struct {
	Propagation MountPropagation `json:",omitempty"`
}

// VolumeOptions represents the options for a mount of type volume.
type VolumeOptions struct {
	NoCopy       bool              `json:",omitempty"`
	Labels       map[string]string `json:",omitempty"`
	DriverConfig *Driver           `json:",omitempty"`
}
