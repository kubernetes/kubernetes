package types

// Pool is used to define a capacity pool.
type Pool struct {

	// Pool unique ID.
	// Read Only: true
	ID string `json:"id"`

	// Pool name.
	// Required: true
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

	// Populated by the system. Read-only.
	CapacityStats CapacityStats `json:"capacityStats"`

	// This field is computed based on NodeSelector value
	// Populated by the system. Read-only.
	Nodes []*Node `json:"nodes"`

	// Labels define a list of labels that describe the pool.
	Labels map[string]string `json:"labels"`
}

// Pools is a collection of Pool objects
type Pools []*Pool
