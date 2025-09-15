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
	"net/url"
	"strings"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/internal"
	jsonv2 "k8s.io/kube-openapi/pkg/internal/third_party/go-json-experiment/json"
)

// BooleanProperty creates a boolean property
func BooleanProperty() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"boolean"}}}
}

// BoolProperty creates a boolean property
func BoolProperty() *Schema { return BooleanProperty() }

// StringProperty creates a string property
func StringProperty() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"string"}}}
}

// CharProperty creates a string property
func CharProperty() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"string"}}}
}

// Float64Property creates a float64/double property
func Float64Property() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"number"}, Format: "double"}}
}

// Float32Property creates a float32/float property
func Float32Property() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"number"}, Format: "float"}}
}

// Int8Property creates an int8 property
func Int8Property() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"integer"}, Format: "int8"}}
}

// Int16Property creates an int16 property
func Int16Property() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"integer"}, Format: "int16"}}
}

// Int32Property creates an int32 property
func Int32Property() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"integer"}, Format: "int32"}}
}

// Int64Property creates an int64 property
func Int64Property() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"integer"}, Format: "int64"}}
}

// StrFmtProperty creates a property for the named string format
func StrFmtProperty(format string) *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"string"}, Format: format}}
}

// DateProperty creates a date property
func DateProperty() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"string"}, Format: "date"}}
}

// DateTimeProperty creates a date time property
func DateTimeProperty() *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"string"}, Format: "date-time"}}
}

// MapProperty creates a map property
func MapProperty(property *Schema) *Schema {
	return &Schema{SchemaProps: SchemaProps{Type: []string{"object"},
		AdditionalProperties: &SchemaOrBool{Allows: true, Schema: property}}}
}

// RefProperty creates a ref property
func RefProperty(name string) *Schema {
	return &Schema{SchemaProps: SchemaProps{Ref: MustCreateRef(name)}}
}

// RefSchema creates a ref property
func RefSchema(name string) *Schema {
	return &Schema{SchemaProps: SchemaProps{Ref: MustCreateRef(name)}}
}

// ArrayProperty creates an array property
func ArrayProperty(items *Schema) *Schema {
	if items == nil {
		return &Schema{SchemaProps: SchemaProps{Type: []string{"array"}}}
	}
	return &Schema{SchemaProps: SchemaProps{Items: &SchemaOrArray{Schema: items}, Type: []string{"array"}}}
}

// ComposedSchema creates a schema with allOf
func ComposedSchema(schemas ...Schema) *Schema {
	s := new(Schema)
	s.AllOf = schemas
	return s
}

// SchemaURL represents a schema url
type SchemaURL string

// MarshalJSON marshal this to JSON
func (r SchemaURL) MarshalJSON() ([]byte, error) {
	if r == "" {
		return []byte("{}"), nil
	}
	v := map[string]interface{}{"$schema": string(r)}
	return json.Marshal(v)
}

// UnmarshalJSON unmarshal this from JSON
func (r *SchemaURL) UnmarshalJSON(data []byte) error {
	var v map[string]interface{}
	if err := json.Unmarshal(data, &v); err != nil {
		return err
	}
	return r.fromMap(v)
}

func (r *SchemaURL) fromMap(v map[string]interface{}) error {
	if v == nil {
		return nil
	}
	if vv, ok := v["$schema"]; ok {
		if str, ok := vv.(string); ok {
			u, err := url.Parse(str)
			if err != nil {
				return err
			}

			*r = SchemaURL(u.String())
		}
	}
	return nil
}

// SchemaProps describes a JSON schema (draft 4)
type SchemaProps struct {
	ID                   string            `json:"id,omitempty"`
	Ref                  Ref               `json:"-"`
	Schema               SchemaURL         `json:"-"`
	Description          string            `json:"description,omitempty"`
	Type                 StringOrArray     `json:"type,omitempty"`
	Nullable             bool              `json:"nullable,omitempty"`
	Format               string            `json:"format,omitempty"`
	Title                string            `json:"title,omitempty"`
	Default              interface{}       `json:"default,omitempty"`
	Maximum              *float64          `json:"maximum,omitempty"`
	ExclusiveMaximum     bool              `json:"exclusiveMaximum,omitempty"`
	Minimum              *float64          `json:"minimum,omitempty"`
	ExclusiveMinimum     bool              `json:"exclusiveMinimum,omitempty"`
	MaxLength            *int64            `json:"maxLength,omitempty"`
	MinLength            *int64            `json:"minLength,omitempty"`
	Pattern              string            `json:"pattern,omitempty"`
	MaxItems             *int64            `json:"maxItems,omitempty"`
	MinItems             *int64            `json:"minItems,omitempty"`
	UniqueItems          bool              `json:"uniqueItems,omitempty"`
	MultipleOf           *float64          `json:"multipleOf,omitempty"`
	Enum                 []interface{}     `json:"enum,omitempty"`
	MaxProperties        *int64            `json:"maxProperties,omitempty"`
	MinProperties        *int64            `json:"minProperties,omitempty"`
	Required             []string          `json:"required,omitempty"`
	Items                *SchemaOrArray    `json:"items,omitempty"`
	AllOf                []Schema          `json:"allOf,omitempty"`
	OneOf                []Schema          `json:"oneOf,omitempty"`
	AnyOf                []Schema          `json:"anyOf,omitempty"`
	Not                  *Schema           `json:"not,omitempty"`
	Properties           map[string]Schema `json:"properties,omitempty"`
	AdditionalProperties *SchemaOrBool     `json:"additionalProperties,omitempty"`
	PatternProperties    map[string]Schema `json:"patternProperties,omitempty"`
	Dependencies         Dependencies      `json:"dependencies,omitempty"`
	AdditionalItems      *SchemaOrBool     `json:"additionalItems,omitempty"`
	Definitions          Definitions       `json:"definitions,omitempty"`
}

// Marshaling structure only, always edit along with corresponding
// struct (or compilation will fail).
type schemaPropsOmitZero struct {
	ID                   string            `json:"id,omitempty"`
	Ref                  Ref               `json:"-"`
	Schema               SchemaURL         `json:"-"`
	Description          string            `json:"description,omitempty"`
	Type                 StringOrArray     `json:"type,omitzero"`
	Nullable             bool              `json:"nullable,omitzero"`
	Format               string            `json:"format,omitempty"`
	Title                string            `json:"title,omitempty"`
	Default              interface{}       `json:"default,omitzero"`
	Maximum              *float64          `json:"maximum,omitempty"`
	ExclusiveMaximum     bool              `json:"exclusiveMaximum,omitzero"`
	Minimum              *float64          `json:"minimum,omitempty"`
	ExclusiveMinimum     bool              `json:"exclusiveMinimum,omitzero"`
	MaxLength            *int64            `json:"maxLength,omitempty"`
	MinLength            *int64            `json:"minLength,omitempty"`
	Pattern              string            `json:"pattern,omitempty"`
	MaxItems             *int64            `json:"maxItems,omitempty"`
	MinItems             *int64            `json:"minItems,omitempty"`
	UniqueItems          bool              `json:"uniqueItems,omitzero"`
	MultipleOf           *float64          `json:"multipleOf,omitempty"`
	Enum                 []interface{}     `json:"enum,omitempty"`
	MaxProperties        *int64            `json:"maxProperties,omitempty"`
	MinProperties        *int64            `json:"minProperties,omitempty"`
	Required             []string          `json:"required,omitempty"`
	Items                *SchemaOrArray    `json:"items,omitzero"`
	AllOf                []Schema          `json:"allOf,omitempty"`
	OneOf                []Schema          `json:"oneOf,omitempty"`
	AnyOf                []Schema          `json:"anyOf,omitempty"`
	Not                  *Schema           `json:"not,omitzero"`
	Properties           map[string]Schema `json:"properties,omitempty"`
	AdditionalProperties *SchemaOrBool     `json:"additionalProperties,omitzero"`
	PatternProperties    map[string]Schema `json:"patternProperties,omitempty"`
	Dependencies         Dependencies      `json:"dependencies,omitempty"`
	AdditionalItems      *SchemaOrBool     `json:"additionalItems,omitzero"`
	Definitions          Definitions       `json:"definitions,omitempty"`
}

// SwaggerSchemaProps are additional properties supported by swagger schemas, but not JSON-schema (draft 4)
type SwaggerSchemaProps struct {
	Discriminator string                 `json:"discriminator,omitempty"`
	ReadOnly      bool                   `json:"readOnly,omitempty"`
	ExternalDocs  *ExternalDocumentation `json:"externalDocs,omitempty"`
	Example       interface{}            `json:"example,omitempty"`
}

// Marshaling structure only, always edit along with corresponding
// struct (or compilation will fail).
type swaggerSchemaPropsOmitZero struct {
	Discriminator string                 `json:"discriminator,omitempty"`
	ReadOnly      bool                   `json:"readOnly,omitzero"`
	ExternalDocs  *ExternalDocumentation `json:"externalDocs,omitzero"`
	Example       interface{}            `json:"example,omitempty"`
}

// Schema the schema object allows the definition of input and output data types.
// These types can be objects, but also primitives and arrays.
// This object is based on the [JSON Schema Specification Draft 4](http://json-schema.org/)
// and uses a predefined subset of it.
// On top of this subset, there are extensions provided by this specification to allow for more complete documentation.
//
// For more information: http://goo.gl/8us55a#schemaObject
type Schema struct {
	VendorExtensible
	SchemaProps
	SwaggerSchemaProps
	ExtraProps map[string]interface{} `json:"-"`
}

// WithID sets the id for this schema, allows for chaining
func (s *Schema) WithID(id string) *Schema {
	s.ID = id
	return s
}

// WithTitle sets the title for this schema, allows for chaining
func (s *Schema) WithTitle(title string) *Schema {
	s.Title = title
	return s
}

// WithDescription sets the description for this schema, allows for chaining
func (s *Schema) WithDescription(description string) *Schema {
	s.Description = description
	return s
}

// WithProperties sets the properties for this schema
func (s *Schema) WithProperties(schemas map[string]Schema) *Schema {
	s.Properties = schemas
	return s
}

// SetProperty sets a property on this schema
func (s *Schema) SetProperty(name string, schema Schema) *Schema {
	if s.Properties == nil {
		s.Properties = make(map[string]Schema)
	}
	s.Properties[name] = schema
	return s
}

// WithAllOf sets the all of property
func (s *Schema) WithAllOf(schemas ...Schema) *Schema {
	s.AllOf = schemas
	return s
}

// WithMaxProperties sets the max number of properties an object can have
func (s *Schema) WithMaxProperties(max int64) *Schema {
	s.MaxProperties = &max
	return s
}

// WithMinProperties sets the min number of properties an object must have
func (s *Schema) WithMinProperties(min int64) *Schema {
	s.MinProperties = &min
	return s
}

// Typed sets the type of this schema for a single value item
func (s *Schema) Typed(tpe, format string) *Schema {
	s.Type = []string{tpe}
	s.Format = format
	return s
}

// AddType adds a type with potential format to the types for this schema
func (s *Schema) AddType(tpe, format string) *Schema {
	s.Type = append(s.Type, tpe)
	if format != "" {
		s.Format = format
	}
	return s
}

// AsNullable flags this schema as nullable.
func (s *Schema) AsNullable() *Schema {
	s.Nullable = true
	return s
}

// CollectionOf a fluent builder method for an array parameter
func (s *Schema) CollectionOf(items Schema) *Schema {
	s.Type = []string{jsonArray}
	s.Items = &SchemaOrArray{Schema: &items}
	return s
}

// WithDefault sets the default value on this parameter
func (s *Schema) WithDefault(defaultValue interface{}) *Schema {
	s.Default = defaultValue
	return s
}

// WithRequired flags this parameter as required
func (s *Schema) WithRequired(items ...string) *Schema {
	s.Required = items
	return s
}

// AddRequired  adds field names to the required properties array
func (s *Schema) AddRequired(items ...string) *Schema {
	s.Required = append(s.Required, items...)
	return s
}

// WithMaxLength sets a max length value
func (s *Schema) WithMaxLength(max int64) *Schema {
	s.MaxLength = &max
	return s
}

// WithMinLength sets a min length value
func (s *Schema) WithMinLength(min int64) *Schema {
	s.MinLength = &min
	return s
}

// WithPattern sets a pattern value
func (s *Schema) WithPattern(pattern string) *Schema {
	s.Pattern = pattern
	return s
}

// WithMultipleOf sets a multiple of value
func (s *Schema) WithMultipleOf(number float64) *Schema {
	s.MultipleOf = &number
	return s
}

// WithMaximum sets a maximum number value
func (s *Schema) WithMaximum(max float64, exclusive bool) *Schema {
	s.Maximum = &max
	s.ExclusiveMaximum = exclusive
	return s
}

// WithMinimum sets a minimum number value
func (s *Schema) WithMinimum(min float64, exclusive bool) *Schema {
	s.Minimum = &min
	s.ExclusiveMinimum = exclusive
	return s
}

// WithEnum sets a the enum values (replace)
func (s *Schema) WithEnum(values ...interface{}) *Schema {
	s.Enum = append([]interface{}{}, values...)
	return s
}

// WithMaxItems sets the max items
func (s *Schema) WithMaxItems(size int64) *Schema {
	s.MaxItems = &size
	return s
}

// WithMinItems sets the min items
func (s *Schema) WithMinItems(size int64) *Schema {
	s.MinItems = &size
	return s
}

// UniqueValues dictates that this array can only have unique items
func (s *Schema) UniqueValues() *Schema {
	s.UniqueItems = true
	return s
}

// AllowDuplicates this array can have duplicates
func (s *Schema) AllowDuplicates() *Schema {
	s.UniqueItems = false
	return s
}

// AddToAllOf adds a schema to the allOf property
func (s *Schema) AddToAllOf(schemas ...Schema) *Schema {
	s.AllOf = append(s.AllOf, schemas...)
	return s
}

// WithDiscriminator sets the name of the discriminator field
func (s *Schema) WithDiscriminator(discriminator string) *Schema {
	s.Discriminator = discriminator
	return s
}

// AsReadOnly flags this schema as readonly
func (s *Schema) AsReadOnly() *Schema {
	s.ReadOnly = true
	return s
}

// AsWritable flags this schema as writeable (not read-only)
func (s *Schema) AsWritable() *Schema {
	s.ReadOnly = false
	return s
}

// WithExample sets the example for this schema
func (s *Schema) WithExample(example interface{}) *Schema {
	s.Example = example
	return s
}

// WithExternalDocs sets/removes the external docs for/from this schema.
// When you pass empty strings as params the external documents will be removed.
// When you pass non-empty string as one value then those values will be used on the external docs object.
// So when you pass a non-empty description, you should also pass the url and vice versa.
func (s *Schema) WithExternalDocs(description, url string) *Schema {
	if description == "" && url == "" {
		s.ExternalDocs = nil
		return s
	}

	if s.ExternalDocs == nil {
		s.ExternalDocs = &ExternalDocumentation{}
	}
	s.ExternalDocs.Description = description
	s.ExternalDocs.URL = url
	return s
}

// MarshalJSON marshal this to JSON
func (s Schema) MarshalJSON() ([]byte, error) {
	if internal.UseOptimizedJSONMarshaling {
		return internal.DeterministicMarshal(s)
	}
	b1, err := json.Marshal(s.SchemaProps)
	if err != nil {
		return nil, fmt.Errorf("schema props %v", err)
	}
	b2, err := json.Marshal(s.VendorExtensible)
	if err != nil {
		return nil, fmt.Errorf("vendor props %v", err)
	}
	b3, err := s.Ref.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("ref prop %v", err)
	}
	b4, err := s.Schema.MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("schema prop %v", err)
	}
	b5, err := json.Marshal(s.SwaggerSchemaProps)
	if err != nil {
		return nil, fmt.Errorf("common validations %v", err)
	}
	var b6 []byte
	if s.ExtraProps != nil {
		jj, err := json.Marshal(s.ExtraProps)
		if err != nil {
			return nil, fmt.Errorf("extra props %v", err)
		}
		b6 = jj
	}
	return swag.ConcatJSON(b1, b2, b3, b4, b5, b6), nil
}

func (s Schema) MarshalNextJSON(opts jsonv2.MarshalOptions, enc *jsonv2.Encoder) error {
	type ArbitraryKeys map[string]interface{}
	var x struct {
		ArbitraryKeys
		SchemaProps        schemaPropsOmitZero        `json:",inline"`
		SwaggerSchemaProps swaggerSchemaPropsOmitZero `json:",inline"`
		Schema             string                     `json:"$schema,omitempty"`
		Ref                string                     `json:"$ref,omitempty"`
	}
	x.ArbitraryKeys = make(map[string]any, len(s.Extensions)+len(s.ExtraProps))
	for k, v := range s.Extensions {
		if internal.IsExtensionKey(k) {
			x.ArbitraryKeys[k] = v
		}
	}
	for k, v := range s.ExtraProps {
		x.ArbitraryKeys[k] = v
	}
	x.SchemaProps = schemaPropsOmitZero(s.SchemaProps)
	x.SwaggerSchemaProps = swaggerSchemaPropsOmitZero(s.SwaggerSchemaProps)
	x.Ref = s.Ref.String()
	x.Schema = string(s.Schema)
	return opts.MarshalNext(enc, x)
}

// UnmarshalJSON marshal this from JSON
func (s *Schema) UnmarshalJSON(data []byte) error {
	if internal.UseOptimizedJSONUnmarshaling {
		return jsonv2.Unmarshal(data, s)
	}

	props := struct {
		SchemaProps
		SwaggerSchemaProps
	}{}
	if err := json.Unmarshal(data, &props); err != nil {
		return err
	}

	sch := Schema{
		SchemaProps:        props.SchemaProps,
		SwaggerSchemaProps: props.SwaggerSchemaProps,
	}

	var d map[string]interface{}
	if err := json.Unmarshal(data, &d); err != nil {
		return err
	}

	_ = sch.Ref.fromMap(d)
	_ = sch.Schema.fromMap(d)

	delete(d, "$ref")
	delete(d, "$schema")
	for _, pn := range swag.DefaultJSONNameProvider.GetJSONNames(s) {
		delete(d, pn)
	}

	for k, vv := range d {
		lk := strings.ToLower(k)
		if strings.HasPrefix(lk, "x-") {
			if sch.Extensions == nil {
				sch.Extensions = map[string]interface{}{}
			}
			sch.Extensions[k] = vv
			continue
		}
		if sch.ExtraProps == nil {
			sch.ExtraProps = map[string]interface{}{}
		}
		sch.ExtraProps[k] = vv
	}

	*s = sch

	return nil
}

func (s *Schema) UnmarshalNextJSON(opts jsonv2.UnmarshalOptions, dec *jsonv2.Decoder) error {
	var x struct {
		Extensions
		SchemaProps
		SwaggerSchemaProps
	}
	if err := opts.UnmarshalNext(dec, &x); err != nil {
		return err
	}

	if err := x.Ref.fromMap(x.Extensions); err != nil {
		return err
	}

	if err := x.Schema.fromMap(x.Extensions); err != nil {
		return err
	}

	delete(x.Extensions, "$ref")
	delete(x.Extensions, "$schema")

	for _, pn := range swag.DefaultJSONNameProvider.GetJSONNames(s) {
		delete(x.Extensions, pn)
	}
	if len(x.Extensions) == 0 {
		x.Extensions = nil
	}

	s.ExtraProps = x.Extensions.sanitizeWithExtra()
	s.Extensions = internal.SanitizeExtensions(x.Extensions)
	s.SchemaProps = x.SchemaProps
	s.SwaggerSchemaProps = x.SwaggerSchemaProps
	return nil
}
