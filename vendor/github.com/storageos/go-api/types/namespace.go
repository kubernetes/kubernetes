package types

import (
	"context"
	"time"
)

// Namespace is used to as a container to isolate namespace and rule obects.
type Namespace struct {

	// Namespace unique ID.
	// Read Only: true
	ID string `json:"id"`

	// Namespace name.
	// Required: true
	Name string `json:"name"`

	// The optional DisplayName is how the project is displayed in the web console (defaults to name).
	DisplayName string `json:"displayName"`

	// Namespcae description.
	Description string `json:"description"`

	// User-defined key/value metadata.
	Labels map[string]string `json:"labels"`

	// When the namespace was created.
	// Read Only: true
	CreatedAt time.Time `json:"createdAt"`

	// User that created the namespace.
	// Read Only: true
	CreatedBy string `json:"createdBy"`

	// When the namespace was created.
	// Read Only: true
	UpdatedAt time.Time `json:"updatedAt"`
}

// NamespaceCreateOptions are available parameters for creating new namespaces.
type NamespaceCreateOptions struct {

	// Name is the name of the namespace to create.
	// Required: true
	Name string `json:"name"`

	// The optional DisplayName is how the project is displayed in the web console (defaults to name).
	DisplayName string `json:"displayName"`

	// Description describes the namespace.
	Description string `json:"description"`

	// Labels are user-defined key/value metadata.
	Labels map[string]string `json:"labels"`

	// Context can be set with a timeout or can be used to cancel a request.
	Context context.Context `json:"-"`
}
