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
	"k8s.io/kube-openapi/pkg/validation/spec"
)

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
	Explode bool `json:"explode,omitempty"`
	// AllowReserved determines whether the parameter value SHOULD allow reserved characters, as defined by RFC3986
	AllowReserved bool `json:"allowReserved,omitempty"`
}
