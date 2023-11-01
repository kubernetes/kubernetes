/*
Copyright 2018 The Kubernetes Authors.

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

package typed

import (
	"fmt"

	yaml "gopkg.in/yaml.v2"
	"sigs.k8s.io/structured-merge-diff/v4/schema"
	"sigs.k8s.io/structured-merge-diff/v4/value"
)

// YAMLObject is an object encoded in YAML.
type YAMLObject string

// Parser implements YAMLParser and allows introspecting the schema.
type Parser struct {
	Schema schema.Schema
}

// create builds an unvalidated parser.
func create(s YAMLObject) (*Parser, error) {
	p := Parser{}
	err := yaml.Unmarshal([]byte(s), &p.Schema)
	return &p, err
}

func createOrDie(schema YAMLObject) *Parser {
	p, err := create(schema)
	if err != nil {
		panic(fmt.Errorf("failed to create parser: %v", err))
	}
	return p
}

var ssParser = createOrDie(YAMLObject(schema.SchemaSchemaYAML))

// NewParser will build a YAMLParser from a schema. The schema is validated.
func NewParser(schema YAMLObject) (*Parser, error) {
	_, err := ssParser.Type("schema").FromYAML(schema)
	if err != nil {
		return nil, fmt.Errorf("unable to validate schema: %v", err)
	}
	p, err := create(schema)
	if err != nil {
		return nil, err
	}
	return p, nil
}

// TypeNames returns a list of types this parser understands.
func (p *Parser) TypeNames() (names []string) {
	for _, td := range p.Schema.Types {
		names = append(names, td.Name)
	}
	return names
}

// Type returns a helper which can produce objects of the given type. Any
// errors are deferred until a further function is called.
func (p *Parser) Type(name string) ParseableType {
	return ParseableType{
		Schema:  &p.Schema,
		TypeRef: schema.TypeRef{NamedType: &name},
	}
}

// ParseableType allows for easy production of typed objects.
type ParseableType struct {
	TypeRef schema.TypeRef
	Schema  *schema.Schema
}

// IsValid return true if p's schema and typename are valid.
func (p ParseableType) IsValid() bool {
	_, ok := p.Schema.Resolve(p.TypeRef)
	return ok
}

// FromYAML parses a yaml string into an object with the current schema
// and the type "typename" or an error if validation fails.
func (p ParseableType) FromYAML(object YAMLObject, opts ...ValidationOptions) (*TypedValue, error) {
	var v interface{}
	err := yaml.Unmarshal([]byte(object), &v)
	if err != nil {
		return nil, err
	}
	return AsTyped(value.NewValueInterface(v), p.Schema, p.TypeRef, opts...)
}

// FromUnstructured converts a go "interface{}" type, typically an
// unstructured object in Kubernetes world, to a TypedValue. It returns an
// error if the resulting object fails schema validation.
// The provided interface{} must be one of: map[string]interface{},
// map[interface{}]interface{}, []interface{}, int types, float types,
// string or boolean. Nested interface{} must also be one of these types.
func (p ParseableType) FromUnstructured(in interface{}, opts ...ValidationOptions) (*TypedValue, error) {
	return AsTyped(value.NewValueInterface(in), p.Schema, p.TypeRef, opts...)
}

// FromStructured converts a go "interface{}" type, typically an structured object in
// Kubernetes, to a TypedValue. It will return an error if the resulting object fails
// schema validation. The provided "interface{}" value must be a pointer so that the
// value can be modified via reflection. The provided "interface{}" may contain structs
// and types that are converted to Values by the jsonMarshaler interface.
func (p ParseableType) FromStructured(in interface{}, opts ...ValidationOptions) (*TypedValue, error) {
	v, err := value.NewValueReflect(in)
	if err != nil {
		return nil, fmt.Errorf("error creating struct value reflector: %v", err)
	}
	return AsTyped(v, p.Schema, p.TypeRef, opts...)
}

// DeducedParseableType is a ParseableType that deduces the type from
// the content of the object.
var DeducedParseableType ParseableType = createOrDie(YAMLObject(`types:
- name: __untyped_atomic_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
- name: __untyped_deduced_
  scalar: untyped
  list:
    elementType:
      namedType: __untyped_atomic_
    elementRelationship: atomic
  map:
    elementType:
      namedType: __untyped_deduced_
    elementRelationship: separable
`)).Type("__untyped_deduced_")
