package types

import "context"

// PoolCreateOptions are available parameters for creating new pools.
type PoolCreateOptions struct {

	// Pool name.
	// Required: true
	Name string `json:"name"`

	// Pool description.
	Description string `json:"description"`

	// Default determines whether this pool is the default if a volume is
	// provisioned without a pool specified.  There can only be one default pool.
	Default bool `json:"default"`

	// DefaultDriver specifies the storage driver to use by default if there are
	// multiple drivers in the pool and no driver was specified in the
	// provisioning request or assigned by rules.  If no driver was specified and
	// no default set, driver weight is used to determine the default.
	DefaultDriver string `json:"defaultDriver"`

	// ControllerNames is a list of controller names that are participating in the
	// storage pool.
	ControllerNames []string `json:"controllerNames"`

	// DriverNames is a list of backend storage drivers that are available in the
	// storage pool.
	DriverNames []string `json:"driverNames"`

	// Flag describing whether the template is active.
	// Default: false
	Active bool `json:"active"`

	// Labels define a list of labels that describe the pool.
	Labels map[string]string `json:"labels"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
