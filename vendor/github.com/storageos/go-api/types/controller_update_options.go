package types

import "context"

// ControllerUpdateOptions are available parameters for updating existing controllers.
type ControllerUpdateOptions struct {

	// Controller unique ID.
	// Read Only: true
	ID string `json:"id"`

	// Controller name.
	// Read Only: true
	Name string `json:"name"`

	// Description of the controller.
	Description string `json:"description"`

	// Labels are user-defined key/value metadata.
	Labels map[string]string `json:"labels"`

	// Cordon sets the controler into an unschedulable state if true
	Cordon bool `json:"unschedulable"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
