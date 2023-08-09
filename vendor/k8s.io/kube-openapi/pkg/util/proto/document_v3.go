/*
Copyright 2022 The Kubernetes Authors.

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

package proto

import (
	"fmt"
	"reflect"
	"strings"

	openapi_v3 "github.com/google/gnostic/openapiv3"
	"gopkg.in/yaml.v3"
)

// Temporary parse implementation to be used until gnostic->kube-openapi conversion
// is possible.
func NewOpenAPIV3Data(doc *openapi_v3.Document) (Models, error) {
	definitions := Definitions{
		models: map[string]Schema{},
	}

	schemas := doc.GetComponents().GetSchemas()
	if schemas == nil {
		return &definitions, nil
	}

	// Save the list of all models first. This will allow us to
	// validate that we don't have any dangling reference.
	for _, namedSchema := range schemas.GetAdditionalProperties() {
		definitions.models[namedSchema.GetName()] = nil
	}

	// Now, parse each model. We can validate that references exists.
	for _, namedSchema := range schemas.GetAdditionalProperties() {
		path := NewPath(namedSchema.GetName())
		val := namedSchema.GetValue()

		if val == nil {
			continue
		}

		if schema, err := definitions.ParseV3SchemaOrReference(namedSchema.GetValue(), &path); err != nil {
			return nil, err
		} else if schema != nil {
			// Schema may be nil if we hit incompleteness in the conversion,
			// but not a fatal error
			definitions.models[namedSchema.GetName()] = schema
		}
	}

	return &definitions, nil
}

func (d *Definitions) ParseV3SchemaReference(s *openapi_v3.Reference, path *Path) (Schema, error) {
	base := &BaseSchema{
		Description: s.Description,
	}

	if !strings.HasPrefix(s.GetXRef(), "#/components/schemas") {
		// Only resolve references to components/schemas. We may add support
		// later for other in-spec paths, but otherwise treat unrecognized
		// refs as arbitrary/unknown values.
		return &Arbitrary{
			BaseSchema: *base,
		}, nil
	}

	reference := strings.TrimPrefix(s.GetXRef(), "#/components/schemas/")
	if _, ok := d.models[reference]; !ok {
		return nil, newSchemaError(path, "unknown model in reference: %q", reference)
	}

	return &Ref{
		BaseSchema: BaseSchema{
			Description: s.Description,
		},
		reference:   reference,
		definitions: d,
	}, nil
}

func (d *Definitions) ParseV3SchemaOrReference(s *openapi_v3.SchemaOrReference, path *Path) (Schema, error) {
	var schema Schema
	var err error

	switch v := s.GetOneof().(type) {
	case *openapi_v3.SchemaOrReference_Reference:
		// Any references stored in #!/components/... are bound to refer
		// to external documents. This API does not support such a
		// feature.
		//
		// In the weird case that this is a reference to a schema that is
		// not external, we attempt to parse anyway
		schema, err = d.ParseV3SchemaReference(v.Reference, path)
	case *openapi_v3.SchemaOrReference_Schema:
		schema, err = d.ParseSchemaV3(v.Schema, path)
	default:
		panic("unexpected type")
	}

	return schema, err
}

// ParseSchema creates a walkable Schema from an openapi v3 schema. While
// this function is public, it doesn't leak through the interface.
func (d *Definitions) ParseSchemaV3(s *openapi_v3.Schema, path *Path) (Schema, error) {
	switch s.GetType() {
	case object:
		for _, extension := range s.GetSpecificationExtension() {
			if extension.Name == "x-kuberentes-group-version-kind" {
				// Objects with x-kubernetes-group-version-kind are always top
				// level types.
				return d.parseV3Kind(s, path)
			}
		}

		if len(s.GetProperties().GetAdditionalProperties()) > 0 {
			return d.parseV3Kind(s, path)
		}
		return d.parseV3Map(s, path)
	case array:
		return d.parseV3Array(s, path)
	case String, Number, Integer, Boolean:
		return d.parseV3Primitive(s, path)
	default:
		return d.parseV3Arbitrary(s, path)
	}
}

func (d *Definitions) parseV3Kind(s *openapi_v3.Schema, path *Path) (Schema, error) {
	if s.GetType() != object {
		return nil, newSchemaError(path, "invalid object type")
	} else if s.GetProperties() == nil {
		return nil, newSchemaError(path, "object doesn't have properties")
	}

	fields := map[string]Schema{}
	fieldOrder := []string{}

	for _, namedSchema := range s.GetProperties().GetAdditionalProperties() {
		var err error
		name := namedSchema.GetName()
		path := path.FieldPath(name)
		fields[name], err = d.ParseV3SchemaOrReference(namedSchema.GetValue(), &path)
		if err != nil {
			return nil, err
		}
		fieldOrder = append(fieldOrder, name)
	}

	base, err := d.parseV3BaseSchema(s, path)
	if err != nil {
		return nil, err
	}

	return &Kind{
		BaseSchema:     *base,
		RequiredFields: s.GetRequired(),
		Fields:         fields,
		FieldOrder:     fieldOrder,
	}, nil
}

func (d *Definitions) parseV3Arbitrary(s *openapi_v3.Schema, path *Path) (Schema, error) {
	base, err := d.parseV3BaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Arbitrary{
		BaseSchema: *base,
	}, nil
}

func (d *Definitions) parseV3Primitive(s *openapi_v3.Schema, path *Path) (Schema, error) {
	switch s.GetType() {
	case String: // do nothing
	case Number: // do nothing
	case Integer: // do nothing
	case Boolean: // do nothing
	default:
		// Unsupported primitive type. Treat as arbitrary type
		return d.parseV3Arbitrary(s, path)
	}

	base, err := d.parseV3BaseSchema(s, path)
	if err != nil {
		return nil, err
	}

	return &Primitive{
		BaseSchema: *base,
		Type:       s.GetType(),
		Format:     s.GetFormat(),
	}, nil
}

func (d *Definitions) parseV3Array(s *openapi_v3.Schema, path *Path) (Schema, error) {
	if s.GetType() != array {
		return nil, newSchemaError(path, `array should have type "array"`)
	} else if len(s.GetItems().GetSchemaOrReference()) != 1 {
		// This array can have multiple types in it (or no types at all)
		// This is not supported by this conversion.
		// Just return an arbitrary type
		return d.parseV3Arbitrary(s, path)
	}

	sub, err := d.ParseV3SchemaOrReference(s.GetItems().GetSchemaOrReference()[0], path)
	if err != nil {
		return nil, err
	}

	base, err := d.parseV3BaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Array{
		BaseSchema: *base,
		SubType:    sub,
	}, nil
}

// We believe the schema is a map, verify and return a new schema
func (d *Definitions) parseV3Map(s *openapi_v3.Schema, path *Path) (Schema, error) {
	if s.GetType() != object {
		return nil, newSchemaError(path, "invalid object type")
	}
	var sub Schema

	switch p := s.GetAdditionalProperties().GetOneof().(type) {
	case *openapi_v3.AdditionalPropertiesItem_Boolean:
		// What does this boolean even mean?
		base, err := d.parseV3BaseSchema(s, path)
		if err != nil {
			return nil, err
		}
		sub = &Arbitrary{
			BaseSchema: *base,
		}
	case *openapi_v3.AdditionalPropertiesItem_SchemaOrReference:
		if schema, err := d.ParseV3SchemaOrReference(p.SchemaOrReference, path); err != nil {
			return nil, err
		} else {
			sub = schema
		}
	case nil:
		// no subtype?
		sub = &Arbitrary{}
	default:
		panic("unrecognized type " + reflect.TypeOf(p).Name())
	}

	base, err := d.parseV3BaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Map{
		BaseSchema: *base,
		SubType:    sub,
	}, nil
}

func parseV3Interface(def *yaml.Node) (interface{}, error) {
	if def == nil {
		return nil, nil
	}
	var i interface{}
	if err := def.Decode(&i); err != nil {
		return nil, err
	}
	return i, nil
}

func (d *Definitions) parseV3BaseSchema(s *openapi_v3.Schema, path *Path) (*BaseSchema, error) {
	if s == nil {
		return nil, fmt.Errorf("cannot initializae BaseSchema from nil")
	}

	def, err := parseV3Interface(s.GetDefault().ToRawInfo())
	if err != nil {
		return nil, err
	}

	return &BaseSchema{
		Description: s.GetDescription(),
		Default:     def,
		Extensions:  SpecificationExtensionToMap(s.GetSpecificationExtension()),
		Path:        *path,
	}, nil
}

func SpecificationExtensionToMap(e []*openapi_v3.NamedAny) map[string]interface{} {
	values := map[string]interface{}{}

	for _, na := range e {
		if na.GetName() == "" || na.GetValue() == nil {
			continue
		}
		if na.GetValue().GetYaml() == "" {
			continue
		}
		var value interface{}
		err := yaml.Unmarshal([]byte(na.GetValue().GetYaml()), &value)
		if err != nil {
			continue
		}

		values[na.GetName()] = value
	}

	return values
}
