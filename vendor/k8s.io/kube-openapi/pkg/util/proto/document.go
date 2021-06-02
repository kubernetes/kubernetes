/*
Copyright 2017 The Kubernetes Authors.

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
	"sort"
	"strings"

	openapi_v2 "github.com/googleapis/gnostic/openapiv2"
	"gopkg.in/yaml.v2"
)

func newSchemaError(path *Path, format string, a ...interface{}) error {
	err := fmt.Sprintf(format, a...)
	if path.Len() == 0 {
		return fmt.Errorf("SchemaError: %v", err)
	}
	return fmt.Errorf("SchemaError(%v): %v", path, err)
}

// VendorExtensionToMap converts openapi VendorExtension to a map.
func VendorExtensionToMap(e []*openapi_v2.NamedAny) map[string]interface{} {
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

// Definitions is an implementation of `Models`. It looks for
// models in an openapi Schema.
type Definitions struct {
	models map[string]Schema
}

var _ Models = &Definitions{}

// NewOpenAPIData creates a new `Models` out of the openapi document.
func NewOpenAPIData(doc *openapi_v2.Document) (Models, error) {
	definitions := Definitions{
		models: map[string]Schema{},
	}

	// Save the list of all models first. This will allow us to
	// validate that we don't have any dangling reference.
	for _, namedSchema := range doc.GetDefinitions().GetAdditionalProperties() {
		definitions.models[namedSchema.GetName()] = nil
	}

	// Now, parse each model. We can validate that references exists.
	for _, namedSchema := range doc.GetDefinitions().GetAdditionalProperties() {
		path := NewPath(namedSchema.GetName())
		schema, err := definitions.ParseSchema(namedSchema.GetValue(), &path)
		if err != nil {
			return nil, err
		}
		definitions.models[namedSchema.GetName()] = schema
	}

	return &definitions, nil
}

// We believe the schema is a reference, verify that and returns a new
// Schema
func (d *Definitions) parseReference(s *openapi_v2.Schema, path *Path) (Schema, error) {
	// TODO(wrong): a schema with a $ref can have properties. We can ignore them (would be incomplete), but we cannot return an error.
	if len(s.GetProperties().GetAdditionalProperties()) > 0 {
		return nil, newSchemaError(path, "unallowed embedded type definition")
	}
	// TODO(wrong): a schema with a $ref can have a type. We can ignore it (would be incomplete), but we cannot return an error.
	if len(s.GetType().GetValue()) > 0 {
		return nil, newSchemaError(path, "definition reference can't have a type")
	}

	// TODO(wrong): $refs outside of the definitions are completely valid. We can ignore them (would be incomplete), but we cannot return an error.
	if !strings.HasPrefix(s.GetXRef(), "#/definitions/") {
		return nil, newSchemaError(path, "unallowed reference to non-definition %q", s.GetXRef())
	}
	reference := strings.TrimPrefix(s.GetXRef(), "#/definitions/")
	if _, ok := d.models[reference]; !ok {
		return nil, newSchemaError(path, "unknown model in reference: %q", reference)
	}
	base, err := d.parseBaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Ref{
		BaseSchema:  base,
		reference:   reference,
		definitions: d,
	}, nil
}

func parseDefault(def *openapi_v2.Any) (interface{}, error) {
	if def == nil {
		return nil, nil
	}
	var i interface{}
	if err := yaml.Unmarshal([]byte(def.Yaml), &i); err != nil {
		return nil, err
	}
	return i, nil
}

func (d *Definitions) parseBaseSchema(s *openapi_v2.Schema, path *Path) (BaseSchema, error) {
	def, err := parseDefault(s.GetDefault())
	if err != nil {
		return BaseSchema{}, err
	}
	return BaseSchema{
		Description: s.GetDescription(),
		Default:     def,
		Extensions:  VendorExtensionToMap(s.GetVendorExtension()),
		Path:        *path,
	}, nil
}

// We believe the schema is a map, verify and return a new schema
func (d *Definitions) parseMap(s *openapi_v2.Schema, path *Path) (Schema, error) {
	if len(s.GetType().GetValue()) != 0 && s.GetType().GetValue()[0] != object {
		return nil, newSchemaError(path, "invalid object type")
	}
	var sub Schema
	// TODO(incomplete): this misses the boolean case as AdditionalProperties is a bool+schema sum type.
	if s.GetAdditionalProperties().GetSchema() == nil {
		base, err := d.parseBaseSchema(s, path)
		if err != nil {
			return nil, err
		}
		sub = &Arbitrary{
			BaseSchema: base,
		}
	} else {
		var err error
		sub, err = d.ParseSchema(s.GetAdditionalProperties().GetSchema(), path)
		if err != nil {
			return nil, err
		}
	}
	base, err := d.parseBaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Map{
		BaseSchema: base,
		SubType:    sub,
	}, nil
}

func (d *Definitions) parsePrimitive(s *openapi_v2.Schema, path *Path) (Schema, error) {
	var t string
	if len(s.GetType().GetValue()) > 1 {
		return nil, newSchemaError(path, "primitive can't have more than 1 type")
	}
	if len(s.GetType().GetValue()) == 1 {
		t = s.GetType().GetValue()[0]
	}
	switch t {
	case String: // do nothing
	case Number: // do nothing
	case Integer: // do nothing
	case Boolean: // do nothing
	// TODO(wrong): this misses "null". Would skip the null case (would be incomplete), but we cannot return an error.
	default:
		return nil, newSchemaError(path, "Unknown primitive type: %q", t)
	}
	base, err := d.parseBaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Primitive{
		BaseSchema: base,
		Type:       t,
		Format:     s.GetFormat(),
	}, nil
}

func (d *Definitions) parseArray(s *openapi_v2.Schema, path *Path) (Schema, error) {
	if len(s.GetType().GetValue()) != 1 {
		return nil, newSchemaError(path, "array should have exactly one type")
	}
	if s.GetType().GetValue()[0] != array {
		return nil, newSchemaError(path, `array should have type "array"`)
	}
	if len(s.GetItems().GetSchema()) != 1 {
		// TODO(wrong): Items can have multiple elements. We can ignore Items then (would be incomplete), but we cannot return an error.
		// TODO(wrong): "type: array" witohut any items at all is completely valid.
		return nil, newSchemaError(path, "array should have exactly one sub-item")
	}
	sub, err := d.ParseSchema(s.GetItems().GetSchema()[0], path)
	if err != nil {
		return nil, err
	}
	base, err := d.parseBaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Array{
		BaseSchema: base,
		SubType:    sub,
	}, nil
}

func (d *Definitions) parseKind(s *openapi_v2.Schema, path *Path) (Schema, error) {
	if len(s.GetType().GetValue()) != 0 && s.GetType().GetValue()[0] != object {
		return nil, newSchemaError(path, "invalid object type")
	}
	if s.GetProperties() == nil {
		return nil, newSchemaError(path, "object doesn't have properties")
	}

	fields := map[string]Schema{}
	fieldOrder := []string{}

	for _, namedSchema := range s.GetProperties().GetAdditionalProperties() {
		var err error
		name := namedSchema.GetName()
		path := path.FieldPath(name)
		fields[name], err = d.ParseSchema(namedSchema.GetValue(), &path)
		if err != nil {
			return nil, err
		}
		fieldOrder = append(fieldOrder, name)
	}

	base, err := d.parseBaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Kind{
		BaseSchema:     base,
		RequiredFields: s.GetRequired(),
		Fields:         fields,
		FieldOrder:     fieldOrder,
	}, nil
}

func (d *Definitions) parseArbitrary(s *openapi_v2.Schema, path *Path) (Schema, error) {
	base, err := d.parseBaseSchema(s, path)
	if err != nil {
		return nil, err
	}
	return &Arbitrary{
		BaseSchema: base,
	}, nil
}

// ParseSchema creates a walkable Schema from an openapi schema. While
// this function is public, it doesn't leak through the interface.
func (d *Definitions) ParseSchema(s *openapi_v2.Schema, path *Path) (Schema, error) {
	if s.GetXRef() != "" {
		// TODO(incomplete): ignoring the rest of s is wrong. As long as there are no conflict, everything from s must be considered
		// Reference: https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#path-item-object
		return d.parseReference(s, path)
	}
	objectTypes := s.GetType().GetValue()
	switch len(objectTypes) {
	case 0:
		// in the OpenAPI schema served by older k8s versions, object definitions created from structs did not include
		// the type:object property (they only included the "properties" property), so we need to handle this case
		// TODO: validate that we ever published empty, non-nil properties. JSON roundtripping nils them.
		if s.GetProperties() != nil {
			// TODO(wrong): when verifying a non-object later against this, it will be rejected as invalid type.
			// TODO(CRD validation schema publishing): we have to filter properties (empty or not) if type=object is not given
			return d.parseKind(s, path)
		} else {
			// Definition has no type and no properties. Treat it as an arbitrary value
			// TODO(incomplete): what if it has additionalProperties=false or patternProperties?
			// ANSWER: parseArbitrary is less strict than it has to be with patternProperties (which is ignored). So this is correct (of course not complete).
			return d.parseArbitrary(s, path)
		}
	case 1:
		t := objectTypes[0]
		switch t {
		case object:
			if s.GetProperties() != nil {
				return d.parseKind(s, path)
			} else {
				return d.parseMap(s, path)
			}
		case array:
			return d.parseArray(s, path)
		}
		return d.parsePrimitive(s, path)
	default:
		// the OpenAPI generator never generates (nor it ever did in the past) OpenAPI type definitions with multiple types
		// TODO(wrong): this is rejecting a completely valid OpenAPI spec
		// TODO(CRD validation schema publishing): filter these out
		return nil, newSchemaError(path, "definitions with multiple types aren't supported")
	}
}

// LookupModel is public through the interface of Models. It
// returns a visitable schema from the given model name.
func (d *Definitions) LookupModel(model string) Schema {
	return d.models[model]
}

func (d *Definitions) ListModels() []string {
	models := []string{}

	for model := range d.models {
		models = append(models, model)
	}

	sort.Strings(models)
	return models
}

type Ref struct {
	BaseSchema

	reference   string
	definitions *Definitions
}

var _ Reference = &Ref{}

func (r *Ref) Reference() string {
	return r.reference
}

func (r *Ref) SubSchema() Schema {
	return r.definitions.models[r.reference]
}

func (r *Ref) Accept(v SchemaVisitor) {
	v.VisitReference(r)
}

func (r *Ref) GetName() string {
	return fmt.Sprintf("Reference to %q", r.reference)
}
