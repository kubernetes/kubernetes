// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

import (
	"encoding/json"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
	"k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json/jsontext"
)

const (
	jsonArray = "array"
)

// HeaderProps describes a response header
type HeaderProps struct {
	Description string `json:"description,omitempty"`
}

// Header describes a header for a response of the API
//
// For more information: http://goo.gl/8us55a#headerObject
type Header struct {
	CommonValidations
	SimpleSchema
	VendorExtensible
	HeaderProps
}

// MarshalJSON marshal this to JSON
func (h Header) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(h)
	}
	b1, err := json.Marshal(h.CommonValidations)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(h.SimpleSchema)
	if err != nil {
		return nil, err
	}
	b3, err := json.Marshal(h.HeaderProps)
	if err != nil {
		return nil, err
	}
	b4, err := json.Marshal(h.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2, b3, b4), nil
}

func (h Header) MarshalJSONTo(enc *jsontext.Encoder) error {
	var x struct {
		CommonValidations commonValidationsOmitZero `json:",inline"`
		SimpleSchema      simpleSchemaOmitZero      `json:",inline"`
		Extensions        Extensions                `json:",inline"`
		HeaderProps
	}
	x.CommonValidations = commonValidationsOmitZero(h.CommonValidations)
	x.SimpleSchema = simpleSchemaOmitZero(h.SimpleSchema)
	x.Extensions = internal.SanitizeExtensions(h.Extensions)
	x.HeaderProps = h.HeaderProps
	return jsonv2.MarshalEncode(enc, x)
}

// UnmarshalJSON unmarshals this header from JSON
func (h *Header) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, h)
	}

	if err := json.Unmarshal(data, &h.CommonValidations); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &h.SimpleSchema); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &h.VendorExtensible); err != nil {
		return err
	}
	return json.Unmarshal(data, &h.HeaderProps)
}

func (h *Header) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var x struct {
		CommonValidations
		SimpleSchema
		Extensions Extensions `json:",inline"`
		HeaderProps
	}

	if err := jsonv2.UnmarshalDecode(dec, &x); err != nil {
		return err
	}

	h.CommonValidations = x.CommonValidations
	h.SimpleSchema = x.SimpleSchema
	h.Extensions = internal.SanitizeExtensions(x.Extensions)
	h.HeaderProps = x.HeaderProps

	return nil
}
