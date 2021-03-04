package types

import "context"

// PoolOptions are available parameters for creating or updating pools.
type PoolOptions struct {
	ID   string `json:"id"`
	Name string `json:"name"`

	// Pool description.
	Description string `json:"description"`

	// Default determines whether this pool is the default if a volume is
	// provisioned without a pool specified.  There can only be one default pool.
	Default bool `json:"default"`

	NodeSelector string `json:"nodeSelector"`

	// DeviceSelector - specifies a selector to filter node devices based on their labels.
	// Only devices from nodes that are in the 'NodeNames' list can be selected
	DeviceSelector string `json:"deviceSelector"`

	// Labels define a list of labels that describe the pool.
	Labels map[string]string `json:"labels"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
