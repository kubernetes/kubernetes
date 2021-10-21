package spec3

import (
	"encoding/json"
	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// MediaType a struct that allows you to specify content format, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#mediaTypeObject
//
// Note that this struct is actually a thin wrapper around MediaTypeProps to make it referable and extensibl
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

// MediaTypeProps a struct that allows you to specify content format, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#mediaTypeObject
type MediaTypeProps struct {
	// Schema holds the schema defining the type used for the media type
	Schema    *spec.Schema `json:"schema,omitempty"`
	// the following fields are missing:
	// TODO: Example field is missing - (example	Any	Example of the media type. The example object SHOULD be in the correct format as specified by the media type. The example object is mutually exclusive of the examples object. Furthermore, if referencing a schema which contains an example, the example value SHALL override the example provided by the schema.)
	// TODO: Examples field is missing - (examples	Map[ string, Example Object | Reference Object]	Examples of the media type. Each example object SHOULD match the media type and specified schema if present. The examples object is mutually exclusive of the example object. Furthermore, if referencing a schema which contains an example, the examples value SHALL override the example provided by the schema.)
	// TODO: Encoding field is missing - (encoding	Map[string, Encoding Object]	A map between a property name and its encoding information. The key, being the property name, MUST exist in the schema as a property. The encoding object SHALL only apply to requestBody objects when the media type is multipart or application/x-www-form-urlencoded.)
}
