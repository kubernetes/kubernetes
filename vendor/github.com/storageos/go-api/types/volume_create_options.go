package types

import "context"

// VolumeCreateOptions are available parameters for creating new volumes.
type VolumeCreateOptions struct {

	// Name is the name of the volume to create.
	// Required: true
	Name string `json:"name"`

	// Description describes the volume.
	Description string `json:"description"`

	// Size in GB.
	// Required: true
	Size int `json:"size"`

	// Pool is the name or id of capacity pool to provision the volume in.
	Pool string `json:"pool"`

	// Filesystem type to mount.  May be set on create, or set by rules to influence client.
	FSType string `json:"fsType"`

	// Namespace is the object scope, such as for teams and projects.
	Namespace string `json:"namespace"`

	// Labels are user-defined key/value metadata.
	Labels map[string]string `json:"labels"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
