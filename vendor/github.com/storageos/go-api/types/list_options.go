package types

import "context"

// ListOptions are optional parameters for finding and listing most objects.
type ListOptions struct {

	// FieldSelector restricts the list of returned objects by their fields. Defaults to everything.
	FieldSelector string

	// LabelSelector restricts the list of returned objects by their labels. Defaults to everything.
	LabelSelector string

	// Namespace is the object scope, such as for teams and projects.
	Namespace string

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context
}
