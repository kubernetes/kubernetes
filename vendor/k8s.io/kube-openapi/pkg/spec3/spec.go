package spec3

import (
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// OpenAPI is an object that describes an API and conforms to the OpenAPI Specification.
//
// Note: at the moment this struct doesn't fully conforms to the OpenAPI Specification in version 3.0,
//       it is just a proof of concept
type OpenAPI struct {
	// Version represents the semantic version number of the OpenAPI Specification that this document uses
	Version string `json:"openapi"`

	// Info provides metadata about the API
	Info *spec.Info `json:"info"`

	// Paths holds the available target and operations for the API
	Paths *Paths `json:"paths,omitempty"`

	// Components hold various schemas for the specification
	Components *Components `json:"components,omitempty"`
}
