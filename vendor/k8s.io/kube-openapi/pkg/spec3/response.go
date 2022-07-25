/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package spec3

import (
	"encoding/json"
	"strconv"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"github.com/go-openapi/swag"
)

// Responses holds the list of possible responses as they are returned from executing this operation
//
// Note that this struct is actually a thin wrapper around ResponsesProps to make it referable and extensible
type Responses struct {
	ResponsesProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Responses as JSON
func (r *Responses) MarshalJSON() ([]byte, error) {
	b1, err := json.Marshal(r.ResponsesProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(r.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

func (r *Responses) UnmarshalJSON(data []byte) error {
	if err := json.Unmarshal(data, &r.ResponsesProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &r.VendorExtensible); err != nil {
		return err
	}

	return nil
}

// ResponsesProps holds the list of possible responses as they are returned from executing this operation
type ResponsesProps struct {
	// Default holds the documentation of responses other than the ones declared for specific HTTP response codes. Use this field to cover undeclared responses
	Default *Response `json:"-"`
	// StatusCodeResponses holds a map of any HTTP status code to the response definition
	StatusCodeResponses map[int]*Response `json:"-"`
}

// MarshalJSON is a custom marshal function that knows how to encode ResponsesProps as JSON
func (r ResponsesProps) MarshalJSON() ([]byte, error) {
	toser := map[string]*Response{}
	if r.Default != nil {
		toser["default"] = r.Default
	}
	for k, v := range r.StatusCodeResponses {
		toser[strconv.Itoa(k)] = v
	}
	return json.Marshal(toser)
}

// UnmarshalJSON unmarshals responses from JSON
func (r *ResponsesProps) UnmarshalJSON(data []byte) error {
	var res map[string]*Response
	if err := json.Unmarshal(data, &res); err != nil {
		return nil
	}
	if v, ok := res["default"]; ok {
		r.Default = v
		delete(res, "default")
	}
	for k, v := range res {
		if nk, err := strconv.Atoi(k); err == nil {
			if r.StatusCodeResponses == nil {
				r.StatusCodeResponses = map[int]*Response{}
			}
			r.StatusCodeResponses[nk] = v
		}
	}
	return nil
}

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


// Link represents a possible design-time link for a response, more at https://swagger.io/specification/#link-object
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
	// OperationId is the name of an existing, resolvable OAS operation
	OperationId string `json:"operationId,omitempty"`
	// Parameters is a map representing parameters to pass to an operation as specified with operationId or identified via operationRef
	Parameters map[string]interface{} `json:"parameters,omitempty"`
	// Description holds a description of the link
	Description string `json:"description,omitempty"`
	// RequestBody is a literal value or expresion to use as a request body when calling the target operation
	RequestBody interface{} `json:"requestBody,omitempty"`
	// Server holds a server object used by the target operation
	Server *Server `json:"server,omitempty"`
}
