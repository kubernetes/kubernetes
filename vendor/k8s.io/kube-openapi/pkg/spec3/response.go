package spec3

import (
	"encoding/json"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"github.com/go-openapi/swag"
)

// Response describes a single response from an API Operation, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#responseObject
//
// Note that this struct is actually a thin wrapper around ResponseProps to make it referable and extensible
type Response struct {
	spec.Refable
	ResponseProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Response as JSON
func (r *Response) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(r.Refable)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(r.ResponseProps)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(r.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

func (r *Response) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &r.Refable); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.ResponseProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.VendorExtensible); err != nil {
		return err
	}

	return nil
}

// ResponseProps describes a single response from an API Operation, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#responseObject
type ResponseProps struct {
	// Description holds a short description of the response
	Description string `json:"description,omitempty"`
	// Headers holds a maps of a headers name to its definition
	Headers map[string]*Header `json:"headers,omitempty"`
	// Content holds a map containing descriptions of potential response payloads
	Content map[string]*MediaType `json:"content,omitempty"`
	// Links is a map of operations links that can be followed from the response
	Links map[string]*Link `json:"links,omitempty"`
}


type Link struct {
	spec.Refable
	LinkProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Link as JSON
func (r *Link) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(r.Refable)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(r.LinkProps)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(r.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

func (r *Link) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &r.Refable); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.LinkProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.VendorExtensible); err != nil {
		return err
	}

	return nil
}

// LinkProps describes a single response from an API Operation, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#responseObject
type LinkProps struct {
	OperationId string `json:"operationId,omitempty"`
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	Description string `json:"description,omitempty"`
	RequestBody interface{} `json:"requestBody,omitempty"`
	Server *Server `json:"server,omitempty"`
}
