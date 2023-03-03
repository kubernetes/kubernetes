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
	"fmt"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
)

// Swagger this is the root document object for the API specification.
// It combines what previously was the Resource Listing and API Declaration (version 1.2 and earlier)
// together into one document.
//
// For more information: http://goo.gl/8us55a#swagger-object-
type Swagger struct {
	VendorExtensible
	SwaggerProps
}

// MarshalJSON marshals this swagger structure to json
func (s Swagger) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(s)
	}
	b1, err := json.Marshal(s.SwaggerProps)
	if err != nil {
		return nil, err
	}
	b2, err := json.Marshal(s.VendorExtensible)
	if err != nil {
		return nil, err
	}
	return swag.ConcatJSON(b1, b2), nil
}

// MarshalJSON marshals this swagger structure to json
func (s Swagger) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	var x struct {
		Extensions
		SwaggerProps
	}
	x.Extensions = internal.SanitizeExtensions(s.Extensions)
	x.SwaggerProps = s.SwaggerProps
	return opts.MarshalNext(enc, x)
}

// UnmarshalJSON unmarshals a swagger spec from json
func (s *Swagger) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, s)
	}
	var sw Swagger
	if err := json.Unmarshal(data, &sw.SwaggerProps); err != nil {
		return err
	}
	if err := json.Unmarshal(data, &sw.VendorExtensible); err != nil {
		return err
	}
	*s = sw
	return nil
}

func (s *Swagger) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	// Note: If you're willing to make breaking changes, it is possible to
	// optimize this and other usages of this pattern:
	// https://github.com/kubernetes/kube-openapi/pull/319#discussion_r983165948
	var x struct {
		Extensions
		SwaggerProps
	}

	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}
	s.Extensions = internal.SanitizeExtensions(x.Extensions)
	s.SwaggerProps = x.SwaggerProps
	return nil
}

// SwaggerProps captures the top-level properties of an Api specification
//
// NOTE: validation rules
// - the scheme, when present must be from [http, https, ws, wss]
// - BasePath must start with a leading "/"
// - Paths is required
type SwaggerProps struct {
	ID                  string                 `json:"id,omitempty"`
	Consumes            []string               `json:"consumes,omitempty"`
	Produces            []string               `json:"produces,omitempty"`
	Schemes             []string               `json:"schemes,omitempty"`
	Swagger             string                 `json:"swagger,omitempty"`
	Info                *Info                  `json:"info,omitempty"`
	Host                string                 `json:"host,omitempty"`
	BasePath            string                 `json:"basePath,omitempty"`
	Paths               *Paths                 `json:"paths"`
	Definitions         Definitions            `json:"definitions,omitempty"`
	Parameters          map[string]Parameter   `json:"parameters,omitempty"`
	Responses           map[string]Response    `json:"responses,omitempty"`
	SecurityDefinitions SecurityDefinitions    `json:"securityDefinitions,omitempty"`
	Security            []map[string][]string  `json:"security,omitempty"`
	Tags                []Tag                  `json:"tags,omitempty"`
	ExternalDocs        *ExternalDocumentation `json:"externalDocs,omitempty"`
}

// Dependencies represent a dependencies property
type Dependencies map[string]SchemaOrStringArray

// SchemaOrBool represents a schema or boolean value, is biased towards true for the boolean property
type SchemaOrBool struct {
	Allows bool
	Schema *Schema
}

var jsTrue = []byte("true")
var jsFalse = []byte("false")

// MarshalJSON convert this object to JSON
func (s SchemaOrBool) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(s)
	}
	if s.Schema != nil {
		return json.Marshal(s.Schema)
	}

	if s.Schema == nil && !s.Allows {
		return jsFalse, nil
	}
	return jsTrue, nil
}

// MarshalJSON convert this object to JSON
func (s SchemaOrBool) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	if s.Schema != nil {
		return opts.MarshalNext(enc, s.Schema)
	}

	if s.Schema == nil && !s.Allows {
		return enc.WriteToken(jsonv2.False)
	}
	return enc.WriteToken(jsonv2.True)
}

// UnmarshalJSON converts this bool or schema object from a JSON structure
func (s *SchemaOrBool) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, s)
	}

	var nw SchemaOrBool
	if len(data) > 0 && data[0] == '{' {
		var sch Schema
		if err := json.Unmarshal(data, &sch); err != nil {
			return err
		}
		nw.Schema = &sch
		nw.Allows = true
	} else {
		json.Unmarshal(data, &nw.Allows)
	}
	*s = nw
	return nil
}

func (s *SchemaOrBool) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	switch k := dec.PeekKind(); k {
	case '{':
		err := opts.UnmarshalNext(dec, &s.Schema)
		if err != nil {
			return err
		}
		s.Allows = true
		return nil
	case 't', 'f':
		err := opts.UnmarshalNext(dec, &s.Allows)
		if err != nil {
			return err
		}
		return nil
	default:
		return fmt.Errorf("expected object or bool, not '%v'", k.String())
	}
}

// SchemaOrStringArray represents a schema or a string array
type SchemaOrStringArray struct {
	Schema   *Schema
	Property []string
}

// MarshalJSON converts this schema object or array into JSON structure
func (s SchemaOrStringArray) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(s)
	}
	if len(s.Property) > 0 {
		return json.Marshal(s.Property)
	}
	if s.Schema != nil {
		return json.Marshal(s.Schema)
	}
	return []byte("null"), nil
}

// MarshalJSON converts this schema object or array into JSON structure
func (s SchemaOrStringArray) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	if len(s.Property) > 0 {
		return opts.MarshalNext(enc, s.Property)
	}
	if s.Schema != nil {
		return opts.MarshalNext(enc, s.Schema)
	}
	return enc.WriteToken(jsonv2.Null)
}

// UnmarshalJSON converts this schema object or array from a JSON structure
func (s *SchemaOrStringArray) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, s)
	}

	var first byte
	if len(data) > 1 {
		first = data[0]
	}
	var nw SchemaOrStringArray
	if first == '{' {
		var sch Schema
		if err := json.Unmarshal(data, &sch); err != nil {
			return err
		}
		nw.Schema = &sch
	}
	if first == '[' {
		if err := json.Unmarshal(data, &nw.Property); err != nil {
			return err
		}
	}
	*s = nw
	return nil
}

func (s *SchemaOrStringArray) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	switch dec.PeekKind() {
	case '{':
		return opts.UnmarshalNext(dec, &s.Schema)
	case '[':
		return opts.UnmarshalNext(dec, &s.Property)
	default:
		_, err := dec.ReadValue()
		return err
	}
}

// Definitions contains the models explicitly defined in this spec
// An object to hold data types that can be consumed and produced by operations.
// These data types can be primitives, arrays or models.
//
// For more information: http://goo.gl/8us55a#definitionsObject
type Definitions map[string]Schema

// SecurityDefinitions a declaration of the security schemes available to be used in the specification.
// This does not enforce the security schemes on the operations and only serves to provide
// the relevant details for each scheme.
//
// For more information: http://goo.gl/8us55a#securityDefinitionsObject
type SecurityDefinitions map[string]*SecurityScheme

// StringOrArray represents a value that can either be a string
// or an array of strings. Mainly here for serialization purposes
type StringOrArray []string

// Contains returns true when the value is contained in the slice
func (s StringOrArray) Contains(value string) bool {
	for _, str := range s {
		if str == value {
			return true
		}
	}
	return false
}

// UnmarshalJSON unmarshals this string or array object from a JSON array or JSON string
func (s *StringOrArray) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, s)
	}

	var first byte
	if len(data) > 1 {
		first = data[0]
	}

	if first == '[' {
		var parsed []string
		if err := json.Unmarshal(data, &parsed); err != nil {
			return err
		}
		*s = StringOrArray(parsed)
		return nil
	}

	var single interface{}
	if err := json.Unmarshal(data, &single); err != nil {
		return err
	}
	if single == nil {
		return nil
	}
	switch v := single.(type) {
	case string:
		*s = StringOrArray([]string{v})
		return nil
	default:
		return fmt.Errorf("only string or array is allowed, not %T", single)
	}
}

func (s *StringOrArray) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	switch k := dec.PeekKind(); k {
	case '[':
		*s = StringOrArray{}
		return opts.UnmarshalNext(dec, (*[]string)(s))
	case '"':
		*s = StringOrArray{""}
		return opts.UnmarshalNext(dec, &(*s)[0])
	case 'n':
		// Throw out null token
		_, _ = dec.ReadToken()
		return nil
	default:
		return fmt.Errorf("expected string or array, not '%v'", k.String())
	}
}

// MarshalJSON converts this string or array to a JSON array or JSON string
func (s StringOrArray) MarshalJSON() ([]byte, error) {
	if len(s) == 1 {
		return json.Marshal([]string(s)[0])
	}
	return json.Marshal([]string(s))
}

// SchemaOrArray represents a value that can either be a Schema
// or an array of Schema. Mainly here for serialization purposes
type SchemaOrArray struct {
	Schema  *Schema
	Schemas []Schema
}

// Len returns the number of schemas in this property
func (s SchemaOrArray) Len() int {
	if s.Schema != nil {
		return 1
	}
	return len(s.Schemas)
}

// ContainsType returns true when one of the schemas is of the specified type
func (s *SchemaOrArray) ContainsType(name string) bool {
	if s.Schema != nil {
		return s.Schema.Type != nil && s.Schema.Type.Contains(name)
	}
	return false
}

// MarshalJSON converts this schema object or array into JSON structure
func (s SchemaOrArray) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(s)
	}
	if s.Schemas != nil {
		return json.Marshal(s.Schemas)
	}
	return json.Marshal(s.Schema)
}

// MarshalJSON converts this schema object or array into JSON structure
func (s SchemaOrArray) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	if s.Schemas != nil {
		return opts.MarshalNext(enc, s.Schemas)
	}
	return opts.MarshalNext(enc, s.Schema)
}

// UnmarshalJSON converts this schema object or array from a JSON structure
func (s *SchemaOrArray) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, s)
	}

	var nw SchemaOrArray
	var first byte
	if len(data) > 1 {
		first = data[0]
	}
	if first == '{' {
		var sch Schema
		if err := json.Unmarshal(data, &sch); err != nil {
			return err
		}
		nw.Schema = &sch
	}
	if first == '[' {
		if err := json.Unmarshal(data, &nw.Schemas); err != nil {
			return err
		}
	}
	*s = nw
	return nil
}

func (s *SchemaOrArray) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	switch dec.PeekKind() {
	case '{':
		return opts.UnmarshalNext(dec, &s.Schema)
	case '[':
		return opts.UnmarshalNext(dec, &s.Schemas)
	default:
		_, err := dec.ReadValue()
		return err
	}
}
