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
	"fmt"
	"strconv"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
	"k8s.io/kube-openapi/pkg/validation/spec"
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
	if internal.UseOptimizedJSONMarshalingV3 {
		return internal.DeterministicMarshal(r)
	}
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

func (r Responses) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	type ArbitraryKeys map[string]interface{}
	var x struct {
		ArbitraryKeys
		Default *Response `json:"default,omitzero"`
	}
	x.ArbitraryKeys = make(map[string]any, len(r.Extensions)+len(r.StatusCodeResponses))
	for k, v := range r.Extensions {
		if internal.IsExtensionKey(k) {
			x.ArbitraryKeys[k] = v
		}
	}
	for k, v := range r.StatusCodeResponses {
		x.ArbitraryKeys[strconv.Itoa(k)] = v
	}
	x.Default = r.Default
	return opts.MarshalNext(enc, x)
}

func (r *Responses) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, r)
	}
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
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, r)
	}
	var res map[string]json.RawMessage
	if err := json.Unmarshal(data, &res); err != nil {
		return err
	}
	if v, ok := res["default"]; ok {
		value := Response{}
		if err := json.Unmarshal(v, &value); err != nil {
			return err
		}
		r.Default = &value
		delete(res, "default")
	}
	for k, v := range res {
		// Take all integral keys
		if nk, err := strconv.Atoi(k); err == nil {
			if r.StatusCodeResponses == nil {
				r.StatusCodeResponses = map[int]*Response{}
			}
			value := Response{}
			if err := json.Unmarshal(v, &value); err != nil {
				return err
			}
			r.StatusCodeResponses[nk] = &value
		}
	}
	return nil
}

func (r *Responses) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) (err error) {
	tok, err := dec.ReadToken()
	if err != nil {
		return err
	}
	switch k := tok.Kind(); k {
	case 'n':
		*r = Responses{}
		return nil
	case '{':
		for {
			tok, err := dec.ReadToken()
			if err != nil {
				return err
			}
			if tok.Kind() == '}' {
				return nil
			}
			switch k := tok.String(); {
			case internal.IsExtensionKey(k):
				var ext any
				if err := opts.UnmarshalNext(dec, &ext); err != nil {
					return err
				}

				if r.Extensions == nil {
					r.Extensions = make(map[string]any)
				}
				r.Extensions[k] = ext
			case k == "default":
				resp := Response{}
				if err := opts.UnmarshalNext(dec, &resp); err != nil {
					return err
				}
				r.ResponsesProps.Default = &resp
			default:
				if nk, err := strconv.Atoi(k); err == nil {
					resp := Response{}
					if err := opts.UnmarshalNext(dec, &resp); err != nil {
						return err
					}

					if r.StatusCodeResponses == nil {
						r.StatusCodeResponses = map[int]*Response{}
					}
					r.StatusCodeResponses[nk] = &resp
				}
			}
		}
	default:
		return fmt.Errorf("unknown JSON kind: %v", k)
	}
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
	if internal.UseOptimizedJSONMarshalingV3 {
		return internal.DeterministicMarshal(r)
	}
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

func (r Response) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		Ref string `json:"$ref,omitempty"`
		spec.Extensions
		ResponseProps `json:",inline"`
	}
	x.Ref = r.Refable.Ref.String()
	x.Extensions = internal.SanitizeExtensions(r.Extensions)
	x.ResponseProps = r.ResponseProps
	return opts.MarshalNext(enc, x)
}

func (r *Response) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, r)
	}
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

func (r *Response) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		ResponseProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	if err := internal.JSONRefFromMap(&r.Ref.Ref, x.Extensions); err != nil {
		return err
	}
	r.Extensions = internal.SanitizeExtensions(x.Extensions)
	r.ResponseProps = x.ResponseProps
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
	if internal.UseOptimizedJSONMarshalingV3 {
		return internal.DeterministicMarshal(r)
	}
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

func (r *Link) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		Ref string `json:"$ref,omitempty"`
		spec.Extensions
		LinkProps `json:",inline"`
	}
	x.Ref = r.Refable.Ref.String()
	x.Extensions = internal.SanitizeExtensions(r.Extensions)
	x.LinkProps = r.LinkProps
	return opts.MarshalNext(enc, x)
}

func (r *Link) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, r)
	}
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

func (l *Link) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		LinkProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	if err := internal.JSONRefFromMap(&l.Ref.Ref, x.Extensions); err != nil {
		return err
	}
	l.Extensions = internal.SanitizeExtensions(x.Extensions)
	l.LinkProps = x.LinkProps
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
