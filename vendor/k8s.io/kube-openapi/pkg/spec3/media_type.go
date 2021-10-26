package spec3

import (
	"encoding/json"
	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// MediaType a struct that allows you to specify content format, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#mediaTypeObject
//
// Note that this struct is actually a thin wrapper around MediaTypeProps to make it referable and extensible
type MediaType struct {
	MediaTypeProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode MediaType as JSON
func (m *MediaType) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(m.MediaTypeProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(m.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

func (m *MediaType) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &m.MediaTypeProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &m.VendorExtensible); err != nil {
		return err
	}
	return nil
}

// MediaTypeProps a struct that allows you to specify content format, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#mediaTypeObject
type MediaTypeProps struct {
	// Schema holds the schema defining the type used for the media type
	Schema    *spec.Schema `json:"schema,omitempty"`
	// Example of the media type
	Example interface{} `json:"example,omitempty"`
	// Examples of the media type. Each example object should match the media type and specific schema if present
	Examples map[string]*Example `json:"examples,omitempty"`
	// A map between a property name and its encoding information. The key, being the property name, MUST exist in the schema as a property. The encoding object SHALL only apply to requestBody objects when the media type is multipart or application/x-www-form-urlencoded
	Encoding map[string]*Encoding `json:"encoding,omitempty"`
}

type Encoding struct {
	EncodingProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Encoding as JSON
func (e *Encoding) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(e.EncodingProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(e.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

func (e *Encoding) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &e.EncodingProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &e.VendorExtensible); err != nil {
		return err
	}
	return nil
}

type EncodingProps struct {
	// Content Type for encoding a specific property
	ContentType string `json:"contentType,omitempty"`
	// A map allowing additional information to be provided as headers
	Headers map[string]*Header `json:"headers,omitempty"`
	// Describes how a specific property value will be serialized depending on its type
	Style string `json:"style,omitempty"`
	// When this is true, property values of type array or object generate separate parameters for each value of the array, or key-value-pair of the map. For other types of properties this property has no effect
	Explode string `json:"explode,omitempty"`
	// AllowReserved determines whether the parameter value SHOULD allow reserved characters, as defined by RFC3986
	AllowReserved bool `json:"allowReserved,omitempty"`
}
