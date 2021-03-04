package types

import (
	"context"
	"time"
)

// DefaultNamespace is used when a namespace hasn't been specified.
const DefaultNamespace = "default"

// Volume represents storage volume.
// swagger:model Volume
type Volume struct {

	// Volume unique ID.
	// Read Only: true
	ID string `json:"id"`

	// Block device inode.
	// Read Only: true
	Inode uint32 `json:"inode"`

	// Volume name.
	// Required: true
	Name string `json:"name"`

	// Size in GB.
	// Required: true
	Size int `json:"size"`

	// Name of capacity pool to provision the volume in, or the name of the current pool.
	Pool string `json:"pool"`

	// Filesystem type to mount.  May be set on create, or set by rules to influence client.
	FSType string `json:"fsType"`

	// Volume description.
	Description string `json:"description"`

	// User-defined key/value metadata.
	Labels map[string]string `json:"labels"`

	// Namespace is the object name and authentication scope, such as for teams and projects.
	Namespace string `json:"namespace"`

	// node selector (where volumes should land)
	NodeSelector string `json:"nodeSelector"`

	// Volume deployment information for the master volume.
	// Read Only: true
	Master *Deployment `json:"master,omitempty"`

	// Flag indicating if the volume is mounted and in use.
	// Read Only: true
	Mounted bool `json:"mounted"`

	// MountDevice, where the device is located
	MountDevice string `json:"mountDevice"`

	// Mountpoint, where the volume is mounted
	Mountpoint string `json:"mountpoint"`

	// When the volume was mounted.
	// Read Only: true
	MountedAt time.Time `json:"mountedAt,omitempty"`

	// Reference to the node that has the volume mounted.
	// Read Only: true
	MountedBy string `json:"mountedBy,omitempty"`

	// Volume deployment information for the replica volumes.
	// Read Only: true
	Replicas []*Deployment `json:"replicas"`

	// Volume health, one of: healthy, degraded or dead.
	// Read Only: true
	Health string `json:"health"`

	// Short status, one of: pending, evaluating, deploying, active, unavailable, failed, updating, deleting.
	// Read Only: true
	Status string `json:"status"`

	// Status message explaining current status.
	// Read Only: true
	StatusMessage string `json:"statusMessage"`

	// mkfs performed on new volumes
	MkfsDone   bool      `json:"mkfsDone"`
	MkfsDoneAt time.Time `json:"mkfsDoneAt"`

	// When the volume was created.
	// Read Only: true
	CreatedAt time.Time `json:"createdAt"`

	// User that created the volume.
	// Read Only: true
	CreatedBy string `json:"createdBy"`
}

// VolumeMountOptions - used by clients to inform of volume mount operations.
type VolumeMountOptions struct {

	// Volume unique ID.
	ID string `json:"id"`

	// Name is the name of the volume to mount.
	Name string `json:"name"`

	// Mountpoint, where the volume is mounted
	Mountpoint string `json:"mountpoint"`

	// Filesystem type, optional but expected when mounting raw volume
	FsType string `json:"fsType"`

	// Namespace is the object scope, such as for teams and projects.
	Namespace string `json:"namespace"`

	// The hostname of the client mounting the volume.
	Client string `json:"client"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}

// VolumeUnmountOptions - used by clients to inform of volume mount operations.
type VolumeUnmountOptions struct {

	// Volume unique ID.
	ID string `json:"id"`

	// Name is the name of the volume to unmount.
	Name string `json:"name"`

	// Namespace is the object scope, such as for teams and projects.
	Namespace string `json:"namespace"`

	// The hostname of the client unmounting the volume.  Must match the hostname
	// of the client that registered the mount operation.
	Client string `json:"client"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
