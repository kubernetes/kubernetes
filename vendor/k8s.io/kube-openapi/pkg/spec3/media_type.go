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

// MediaType a struct that allows you to specify content format, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#mediaTypeObject
//
// Note that this struct is actually a thin wrapper around MediaTypeProps to make it referable and extensible
type MediaType struct {
	MediaTypeProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode MediaType as JSON
func (m *MediaType) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshalingV3 {
		return internal.DeterministicMarshal(m)
	}
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

func (e *MediaType) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		MediaTypeProps mediaTypePropsOmitZero `json:",inline"`
		spec.Extensions
	}
	x.Extensions = internal.SanitizeExtensions(e.Extensions)
	x.MediaTypeProps = mediaTypePropsOmitZero(e.MediaTypeProps)
	return opts.MarshalNext(enc, x)
}

func (m *MediaType) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshalingV3 {
		return jsonv2.Unmarshal(data, m)
	}
	if err := json.Unmarshal(data, &m.MediaTypeProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &m.VendorExtensible); err != nil {
		return err
	}
	return nil
}

func (m *MediaType) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		spec.Extensions
		MediaTypeProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	m.Extensions = internal.SanitizeExtensions(x.Extensions)
	m.MediaTypeProps = x.MediaTypeProps

	return nil
}

// MediaTypeProps a struct that allows you to specify content format, more at https://github.com/OAI/OpenAPI-Specification/blob/master/versions/3.0.0.md#mediaTypeObject
type MediaTypeProps struct {
	// Schema holds the schema defining the type used for the media type
	Schema *spec.Schema `json:"schema,omitempty"`
	// Example of the media type
	Example interface{} `json:"example,omitempty"`
	// Examples of the media type. Each example object should match the media type and specific schema if present
	Examples map[string]*Example `json:"examples,omitempty"`
	// A map between a property name and its encoding information. The key, being the property name, MUST exist in the schema as a property. The encoding object SHALL only apply to requestBody objects when the media type is multipart or application/x-www-form-urlencoded
	Encoding map[string]*Encoding `json:"encoding,omitempty"`
}

type mediaTypePropsOmitZero struct {
	Schema   *spec.Schema         `json:"schema,omitzero"`
	Example  interface{}          `json:"example,omitempty"`
	Examples map[string]*Example  `json:"examples,omitempty"`
	Encoding map[string]*Encoding `json:"encoding,omitempty"`
}
