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
	jsonRef = "$ref"
)

// SimpleSchema describe swagger simple schemas for parameters and headers
type SimpleSchema struct {
	Type             string      `json:"type,omitempty"`
	Nullable         bool        `json:"nullable,omitempty"`
	Format           string      `json:"format,omitempty"`
	Items            *Items      `json:"items,omitempty"`
	CollectionFormat string      `json:"collectionFormat,omitempty"`
	Default          interface{} `json:"default,omitempty"`
	Example          interface{} `json:"example,omitempty"`
}

// Marshaling structure only, always edit along with corresponding
// struct (or compilation will fail).
type simpleSchemaOmitZero struct {
	Type             string      `json:"type,omitempty"`
	Nullable         bool        `json:"nullable,omitzero"`
	Format           string      `json:"format,omitempty"`
	Items            *Items      `json:"items,omitzero"`
	CollectionFormat string      `json:"collectionFormat,omitempty"`
	Default          interface{} `json:"default,omitempty"`
	Example          interface{} `json:"example,omitempty"`
}

// CommonValidations describe common JSON-schema validations
type CommonValidations struct {
	Maximum          *float64      `json:"maximum,omitempty"`
	ExclusiveMaximum bool          `json:"exclusiveMaximum,omitempty"`
	Minimum          *float64      `json:"minimum,omitempty"`
	ExclusiveMinimum bool          `json:"exclusiveMinimum,omitempty"`
	MaxLength        *int64        `json:"maxLength,omitempty"`
	MinLength        *int64        `json:"minLength,omitempty"`
	Pattern          string        `json:"pattern,omitempty"`
	MaxItems         *int64        `json:"maxItems,omitempty"`
	MinItems         *int64        `json:"minItems,omitempty"`
	UniqueItems      bool          `json:"uniqueItems,omitempty"`
	MultipleOf       *float64      `json:"multipleOf,omitempty"`
	Enum             []interface{} `json:"enum,omitempty"`
}

// Marshaling structure only, always edit along with corresponding
// struct (or compilation will fail).
type commonValidationsOmitZero struct {
	Maximum          *float64      `json:"maximum,omitempty"`
	ExclusiveMaximum bool          `json:"exclusiveMaximum,omitzero"`
	Minimum          *float64      `json:"minimum,omitempty"`
	ExclusiveMinimum bool          `json:"exclusiveMinimum,omitzero"`
	MaxLength        *int64        `json:"maxLength,omitempty"`
	MinLength        *int64        `json:"minLength,omitempty"`
	Pattern          string        `json:"pattern,omitempty"`
	MaxItems         *int64        `json:"maxItems,omitempty"`
	MinItems         *int64        `json:"minItems,omitempty"`
	UniqueItems      bool          `json:"uniqueItems,omitzero"`
	MultipleOf       *float64      `json:"multipleOf,omitempty"`
	Enum             []interface{} `json:"enum,omitempty"`
}

// Items a limited subset of JSON-Schema's items object.
// It is used by parameter definitions that are not located in "body".
//
// For more information: http://goo.gl/8us55a#items-object
type Items struct {
	Refable
	CommonValidations
	SimpleSchema
	VendorExtensible
}

// UnmarshalJSON hydrates this items instance with the data from JSON
func (i *Items) UnmarshalJSON(data []byte) error {
	return jsonv2.Unmarshal(data, i)
}

func (i *Items) UnmarshalJSONFrom(dec *jsontext.Decoder) error {
	var x struct {
		CommonValidations
		SimpleSchema
		Extensions Extensions `json:",embed"`
	}
	if err := jsonv2.UnmarshalDecode(dec, &x); err != nil {
		return err
	}
	if err := i.Refable.Ref.fromMap(x.Extensions); err != nil {
		return err
	}

	i.CommonValidations = x.CommonValidations
	i.SimpleSchema = x.SimpleSchema
	i.Extensions = internal.SanitizeExtensions(x.Extensions)
	return nil
}

// MarshalJSON converts this items object to JSON
func (i Items) MarshalJSON() ([]byte, error) {
	return internal.DeterministicMarshal(i)
}

func (i Items) MarshalJSONTo(enc *jsontext.Encoder) error {
	var x struct {
		CommonValidations commonValidationsOmitZero `json:",embed"`
		SimpleSchema      simpleSchemaOmitZero      `json:",embed"`
		Ref               string                    `json:"$ref,omitempty"`
		Extensions        Extensions                `json:",embed"`
	}
	x.CommonValidations = commonValidationsOmitZero(i.CommonValidations)
	x.SimpleSchema = simpleSchemaOmitZero(i.SimpleSchema)
	x.Ref = i.Refable.Ref.String()
	x.Extensions = internal.SanitizeExtensions(i.Extensions)
	return jsonv2.MarshalEncode(enc, x)
}
