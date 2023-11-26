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

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// Header a struct that describes a single operation parameter, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#parameterObject
//
// Note that this struct is actually a thin wrapper around HeaderProps to make it referable and extensible
type Header struct {
	spec.Refable
	HeaderProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Header as JSON
func (h *Header) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshalingV3 {
		return internal.DeterministicMarshal(h)
	}
	b1, err := json.Marshal(h.Refable)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(h.HeaderProps)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(h.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3), nil
}

func (h *Header) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		Ref         string              `json:"$ref,omitempty"`
		HeaderProps headerPropsOmitZero `json:",inline"`
		spec.Extensions
	}
	x.Ref = h.Refable.Ref.String()
	x.Extensions = internal.SanitizeExtensions(h.Extensions)
	x.HeaderProps = headerPropsOmitZero(h.HeaderProps)
	return opts.MarshalNext(enc, x)
}

func (h *Header) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, h)
	}
	if err := json.Unmarshal(data, &h.Refable); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &h.HeaderProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &h.VendorExtensible); err != nil {
		return err
	}

	return nil
}

func (h *Header) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		HeaderProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	if err := internal.JSONRefFromMap(&h.Ref.Ref, x.Extensions); err != nil {
		return err
	}
	h.Extensions = internal.SanitizeExtensions(x.Extensions)
	h.HeaderProps = x.HeaderProps
	return nil
}

// HeaderProps a struct that describes a header object
type HeaderProps struct {
	// Description holds a brief description of the parameter
	Description string `json:"description,omitempty"`
	// Required determines whether this parameter is mandatory
	Required bool `json:"required,omitempty"`
	// Deprecated declares this operation to be deprecated
	Deprecated bool `json:"deprecated,omitempty"`
	// AllowEmptyValue sets the ability to pass empty-valued parameters
	AllowEmptyValue bool `json:"allowEmptyValue,omitempty"`
	// Style describes how the parameter value will be serialized depending on the type of the parameter value
	Style string `json:"style,omitempty"`
	// Explode when true, parameter values of type array or object generate separate parameters for each value of the array or key-value pair of the map
	Explode bool `json:"explode,omitempty"`
	// AllowReserved determines whether the parameter value SHOULD allow reserved characters, as defined by RFC3986
	AllowReserved bool `json:"allowReserved,omitempty"`
	// Schema holds the schema defining the type used for the parameter
	Schema *spec.Schema `json:"schema,omitempty"`
	// Content holds a map containing the representations for the parameter
	Content map[string]*MediaType `json:"content,omitempty"`
	// Example of the header
	Example interface{} `json:"example,omitempty"`
	// Examples of the header
	Examples map[string]*Example `json:"examples,omitempty"`
}

// Marshaling structure only, always edit along with corresponding
// struct (or compilation will fail).
type headerPropsOmitZero struct {
	Description     string                `json:"description,omitempty"`
	Required        bool                  `json:"required,omitzero"`
	Deprecated      bool                  `json:"deprecated,omitzero"`
	AllowEmptyValue bool                  `json:"allowEmptyValue,omitzero"`
	Style           string                `json:"style,omitempty"`
	Explode         bool                  `json:"explode,omitzero"`
	AllowReserved   bool                  `json:"allowReserved,omitzero"`
	Schema          *spec.Schema          `json:"schema,omitzero"`
	Content         map[string]*MediaType `json:"content,omitempty"`
	Example         interface{}           `json:"example,omitempty"`
	Examples        map[string]*Example   `json:"examples,omitempty"`
}
