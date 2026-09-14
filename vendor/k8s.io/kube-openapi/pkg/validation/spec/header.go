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
	"encoding/json/jsontext"
	jsonv2 "encoding/json/v2"

	"k8s.io/kube-openapi/pkg/internal"
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
	return internal.DeterministicMarshal(h)
}

func (h Header) MarshalJSONTo(enc *jsontext.Encoder) error {
	var x struct {
		CommonValidations commonValidationsOmitZero `json:",embed"`
		SimpleSchema      simpleSchemaOmitZero      `json:",embed"`
		Extensions        Extensions                `json:",embed"`
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
	return jsonv2.Unmarshal(data, h)
}

func (h *Header) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var x struct {
		CommonValidations
		SimpleSchema
		Extensions Extensions `json:",embed"`
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
