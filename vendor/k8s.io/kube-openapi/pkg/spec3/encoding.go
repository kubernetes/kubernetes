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

type Encoding struct {
	EncodingProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Encoding as JSON
func (e *Encoding) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshalingV3 {
		return internal.DeterministicMarshal(e)
	}
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

func (e *Encoding) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		EncodingProps encodingPropsOmitZero `json:",inline"`
		spec.Extensions
	}
	x.Extensions = internal.SanitizeExtensions(e.Extensions)
	x.EncodingProps = encodingPropsOmitZero(e.EncodingProps)
	return opts.MarshalNext(enc, x)
}

func (e *Encoding) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, e)
	}
	if err := json.Unmarshal(data, &e.EncodingProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &e.VendorExtensible); err != nil {
		return err
	}
	return nil
}

func (e *Encoding) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		EncodingProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}

	e.Extensions = internal.SanitizeExtensions(x.Extensions)
	e.EncodingProps = x.EncodingProps
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
	Explode bool `json:"explode,omitempty"`
	// AllowReserved determines whether the parameter value SHOULD allow reserved characters, as defined by RFC3986
	AllowReserved bool `json:"allowReserved,omitempty"`
}

type encodingPropsOmitZero struct {
	ContentType   string             `json:"contentType,omitempty"`
	Headers       map[string]*Header `json:"headers,omitempty"`
	Style         string             `json:"style,omitempty"`
	Explode       bool               `json:"explode,omitzero"`
	AllowReserved bool               `json:"allowReserved,omitzero"`
}
