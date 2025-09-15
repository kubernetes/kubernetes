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

package validate

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/go-openapi/swag"
	"k8s.io/kube-openapi/pkg/validation/errors"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

var (
	specSchemaType = reflect.TypeOf(&spec.Schema{})
	//specItemsType     = reflect.TypeOf(&spec.Items{})
)

// SchemaValidator validates data against a JSON schema
type SchemaValidator struct {
	Path         string
	in           string
	Schema       *spec.Schema
	validators   []ValueValidator
	Root         interface{}
	KnownFormats strfmt.Registry
	Options      SchemaValidatorOptions
}

// AgainstSchema validates the specified data against the provided schema, using a registry of supported formats.
//
// When no pre-parsed *spec.Schema structure is provided, it uses a JSON schema as default. See example.
func AgainstSchema(schema *spec.Schema, data interface{}, formats strfmt.Registry, options ...Option) error {
	res := NewSchemaValidator(schema, nil, "", formats, options...).Validate(data)
	if res.HasErrors() {
		return errors.CompositeValidationError(res.Errors...)
	}
	return nil
}

// NewSchemaValidator creates a new schema validator.
//
// Panics if the provided schema is invalid.
func NewSchemaValidator(schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, options ...Option) *SchemaValidator {
	if schema == nil {
		return nil
	}

	if rootSchema == nil {
		rootSchema = schema
	}

	if ref := schema.Ref.String(); ref != "" {
		panic(fmt.Sprintf("schema references not supported: %s", ref))
	}

	s := SchemaValidator{
		Path:         root,
		in:           "body",
		Schema:       schema,
		Root:         rootSchema,
		KnownFormats: formats,
		Options:      SchemaValidatorOptions{}}
	for _, o := range options {
		o(&s.Options)
	}

	if s.Options.NewValidatorForIndex == nil {
		s.Options.NewValidatorForIndex = s.NewValidatorForIndex
	}
	if s.Options.NewValidatorForField == nil {
		s.Options.NewValidatorForField = s.NewValidatorForField
	}

	s.validators = []ValueValidator{
		s.typeValidator(),
		s.schemaPropsValidator(),
		s.stringValidator(),
		s.formatValidator(),
		s.numberValidator(),
		s.sliceValidator(),
		s.commonValidator(),
		s.objectValidator(),
	}
	return &s
}

func (s *SchemaValidator) NewValidatorForField(field string, schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, opts ...Option) ValueValidator {
	return NewSchemaValidator(schema, rootSchema, root, formats, opts...)
}

func (s *SchemaValidator) NewValidatorForIndex(index int, schema *spec.Schema, rootSchema interface{}, root string, formats strfmt.Registry, opts ...Option) ValueValidator {
	return NewSchemaValidator(schema, rootSchema, root, formats, opts...)
}

// SetPath sets the path for this schema validator
func (s *SchemaValidator) SetPath(path string) {
	s.Path = path
	for _, v := range s.validators {
		v.SetPath(path)
	}
}

// Applies returns true when this schema validator applies
func (s *SchemaValidator) Applies(source interface{}, kind reflect.Kind) bool {
	_, ok := source.(*spec.Schema)
	return ok
}

// Validate validates the data against the schema
func (s *SchemaValidator) Validate(data interface{}) *Result {
	result := new(Result)
	if s == nil {
		return result
	}

	if data == nil {
		result.Merge(s.validators[0].Validate(data)) // type validator
		result.Merge(s.validators[6].Validate(data)) // common validator
		return result
	}

	tpe := reflect.TypeOf(data)
	kind := tpe.Kind()
	for kind == reflect.Ptr {
		tpe = tpe.Elem()
		kind = tpe.Kind()
	}
	d := data

	if kind == reflect.Struct {
		// NOTE: since reflect retrieves the true nature of types
		// this means that all strfmt types passed here (e.g. strfmt.Datetime, etc..)
		// are converted here to strings, and structs are systematically converted
		// to map[string]interface{}.
		d = swag.ToDynamicJSON(data)
	}

	// TODO: this part should be handed over to type validator
	// Handle special case of json.Number data (number marshalled as string)
	isnumber := s.Schema.Type.Contains(numberType) || s.Schema.Type.Contains(integerType)
	if num, ok := data.(json.Number); ok && isnumber {
		if s.Schema.Type.Contains(integerType) { // avoid lossy conversion
			in, erri := num.Int64()
			if erri != nil {
				result.AddErrors(invalidTypeConversionMsg(s.Path, erri))
				result.Inc()
				return result
			}
			d = in
		} else {
			nf, errf := num.Float64()
			if errf != nil {
				result.AddErrors(invalidTypeConversionMsg(s.Path, errf))
				result.Inc()
				return result
			}
			d = nf
		}

		tpe = reflect.TypeOf(d)
		kind = tpe.Kind()
	}

	for _, v := range s.validators {
		if !v.Applies(s.Schema, kind) {
			debugLog("%T does not apply for %v", v, kind)
			continue
		}

		err := v.Validate(d)
		result.Merge(err)
		result.Inc()
	}
	result.Inc()
	return result
}

func (s *SchemaValidator) typeValidator() ValueValidator {
	return &typeValidator{Type: s.Schema.Type, Nullable: s.Schema.Nullable, Format: s.Schema.Format, In: s.in, Path: s.Path}
}

func (s *SchemaValidator) commonValidator() ValueValidator {
	return &basicCommonValidator{
		Path: s.Path,
		In:   s.in,
		Enum: s.Schema.Enum,
	}
}

func (s *SchemaValidator) sliceValidator() ValueValidator {
	return &schemaSliceValidator{
		Path:            s.Path,
		In:              s.in,
		MaxItems:        s.Schema.MaxItems,
		MinItems:        s.Schema.MinItems,
		UniqueItems:     s.Schema.UniqueItems,
		AdditionalItems: s.Schema.AdditionalItems,
		Items:           s.Schema.Items,
		Root:            s.Root,
		KnownFormats:    s.KnownFormats,
		Options:         s.Options,
	}
}

func (s *SchemaValidator) numberValidator() ValueValidator {
	return &numberValidator{
		Path:             s.Path,
		In:               s.in,
		Default:          s.Schema.Default,
		MultipleOf:       s.Schema.MultipleOf,
		Maximum:          s.Schema.Maximum,
		ExclusiveMaximum: s.Schema.ExclusiveMaximum,
		Minimum:          s.Schema.Minimum,
		ExclusiveMinimum: s.Schema.ExclusiveMinimum,
	}
}

func (s *SchemaValidator) stringValidator() ValueValidator {
	return &stringValidator{
		Path:      s.Path,
		In:        s.in,
		MaxLength: s.Schema.MaxLength,
		MinLength: s.Schema.MinLength,
		Pattern:   s.Schema.Pattern,
	}
}

func (s *SchemaValidator) formatValidator() ValueValidator {
	return &formatValidator{
		Path:         s.Path,
		In:           s.in,
		Format:       s.Schema.Format,
		KnownFormats: s.KnownFormats,
	}
}

func (s *SchemaValidator) schemaPropsValidator() ValueValidator {
	sch := s.Schema
	return newSchemaPropsValidator(s.Path, s.in, sch.AllOf, sch.OneOf, sch.AnyOf, sch.Not, sch.Dependencies, s.Root, s.KnownFormats, s.Options.Options()...)
}

func (s *SchemaValidator) objectValidator() ValueValidator {
	return &objectValidator{
		Path:                 s.Path,
		In:                   s.in,
		MaxProperties:        s.Schema.MaxProperties,
		MinProperties:        s.Schema.MinProperties,
		Required:             s.Schema.Required,
		Properties:           s.Schema.Properties,
		AdditionalProperties: s.Schema.AdditionalProperties,
		PatternProperties:    s.Schema.PatternProperties,
		Root:                 s.Root,
		KnownFormats:         s.KnownFormats,
		Options:              s.Options,
	}
}
