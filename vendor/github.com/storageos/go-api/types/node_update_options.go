package types

import "context"

// NodeUpdateOptions are available parameters for updating existing nodes.
type NodeUpdateOptions struct {

	// Node unique ID.
	// Read Only: true
	ID string `json:"id"`

	// Node name.
	// Read Only: true
	Name string `json:"name"`

	// Description of the node.
	Description string `json:"description"`

	// Labels are user-defined key/value metadata.
	Labels map[string]string `json:"labels"`

	// Cordon marks the node as unschedulable if true
	Cordon bool `json:"cordon"`
	Drain  bool `json:"drain"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
