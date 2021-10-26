package spec3

import (
	"encoding/json"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"github.com/go-openapi/swag"
)

// Example https://swagger.io/specification/#example-object

type Example struct {
	spec.Refable
	ExampleProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode RequestBody as JSON
func (e *Example) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(e.Refable)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(e.ExampleProps)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(e.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

type ExampleProps struct {
	// Summary holds a short description of the example
	Summary string `json:"summary,omitempty"`
	// Description holds a long description of the example
	Description string `json:"description,omitempty"`
	// Embedded literal example.
	Value interface{} `json:"value,omitempty"`
	// A URL that points to the literal example. This provides the capability to reference examples that cannot easily be included in JSON or YAML documents.
	ExternalValue string `json:"externalValue,omitempty"`
}
