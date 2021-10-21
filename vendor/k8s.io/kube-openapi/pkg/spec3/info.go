package spec3

import (
	"encoding/json"

	"github.com/go-openapi/spec"
	"github.com/go-openapi/swag"
)

// Info provides metadata about the API
type Info struct {
	InfoProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Paths as JSON
func (i Info) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(i.InfoProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(i.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// InfoProps provides metadata about the API
type InfoProps struct {
	// Title of the application
	Title string `json:"title"`
	// Description is a short description of the application
	Description string `json:"description,omitempty"`
	// TermsOfService represents a URL to the Terms of Service for the API
	TermsOfService string `json:"termsOfService,omitempty"`
	// The contact information for the exposed API
	Contact *Contact `json:"contact,omitempty"`
	// License the license information for the exposed API
	License License `json:"license"`
	// Version represents the version of the OpenAPI document (which is distinct from the OpenAPI Specification version or the API implementation version)
	Version string `json:"version"`
}

// Contact information for the exposed API
type Contact struct {
	// Name	of the contact person/organization
	Name string `json:"name,omitempty"`
	// Url pointing to the contact information
	Url string `json:"url,omitempty"`
	// Email address of the contact person/organization
	Email string `json:"email,omitempty"`
}

// License information for the exposed API
type License struct {
	// Name of the license used for the API
	Name string `json:"name,omitempty"`
	// URL to the license used for the API
	Url string `json:"url,omitempty"`
}
