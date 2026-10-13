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
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	"k8s.io/kube-openapi/pkg/internal"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

type Encoding struct {
	EncodingProps
	spec.VendorExtensible
}

// MarshalJSON is a custom marshal function that knows how to encode Encoding as JSON
func (e *Encoding) MarshalJSON() ([]byte, error) {
	return internal.DeterministicMarshal(e)
}

func (e *Encoding) MarshalJSONTo(enc *jsontext.Encoder) error {
	var x struct {
		EncodingProps encodingPropsOmitZero `json:",embed"`
		Extensions    spec.Extensions       `json:",embed"`
	}
	x.Extensions = internal.SanitizeExtensions(e.Extensions)
	x.EncodingProps = encodingPropsOmitZero(e.EncodingProps)
	return jsonv2.MarshalEncode(enc, x)
}

func (e *Encoding) UnmarshalJSON(data []byte) error {
	return jsonv2.Unmarshal(data, e)
}

func (e *Encoding) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var x struct {
		Extensions spec.Extensions `json:",embed"`
		EncodingProps
	}
	if err := jsonv2.UnmarshalDecode(dec, &x); err != nil {
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
