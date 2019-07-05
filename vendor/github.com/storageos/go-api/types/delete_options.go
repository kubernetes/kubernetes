package types

import "context"

// DeleteOptions are available parameters for deleting existing volumes.
type DeleteOptions struct {

	// Volume unique ID.
	// Read Only: true
	ID string `json:"id"`

	// Volume name.
	// Read Only: true
	Name string `json:"name"`

	// Namespace is the object scope, such as for teams and projects.
	Namespace string `json:"namespace"`

	// Force will cause the volume to be deleted even if it's in use.
	Force bool `json:"force"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
