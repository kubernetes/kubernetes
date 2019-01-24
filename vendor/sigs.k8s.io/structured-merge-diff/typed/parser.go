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
	"sigs.k8s.io/structured-merge-diff/schema"
	"sigs.k8s.io/structured-merge-diff/value"
)

// YAMLObject is an object encoded in YAML.
type YAMLObject string

// Parser implements YAMLParser and allows introspecting the schema.
type Parser struct {
	Schema schema.Schema
}

// create builds an unvalidated parser.
func create(schema YAMLObject) (*Parser, error) {
	p := Parser{}
	err := yaml.Unmarshal([]byte(schema), &p.Schema)
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
	return create(schema)
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
	return &parseableType{
		parser:   p,
		typename: name,
	}
}

// ParseableType allows for easy production of typed objects.
type ParseableType interface {
	IsValid() bool
	FromYAML(YAMLObject) (TypedValue, error)
	FromUnstructured(interface{}) (TypedValue, error)
}

type parseableType struct {
	parser   *Parser
	typename string
}

var _ ParseableType = &parseableType{}

// IsValid return true if p's schema and typename are valid.
func (p *parseableType) IsValid() bool {
	_, ok := p.parser.Schema.Resolve(schema.TypeRef{NamedType: &p.typename})
	return ok
}

// FromYAML parses a yaml string into an object with the current schema
// and the type "typename" or an error if validation fails.
func (p *parseableType) FromYAML(object YAMLObject) (TypedValue, error) {
	v, err := value.FromYAML([]byte(object))
	if err != nil {
		return nil, err
	}
	return AsTyped(v, &p.parser.Schema, p.typename)
}

// FromUnstructured converts a go interface to a TypedValue. It will return an
// error if the resulting object fails schema validation.
func (p *parseableType) FromUnstructured(in interface{}) (TypedValue, error) {
	v, err := value.FromUnstructured(in)
	if err != nil {
		return nil, err
	}
	return AsTyped(v, &p.parser.Schema, p.typename)
}

// DeducedParseableType is a ParseableType that deduces the type from
// the content of the object.
type DeducedParseableType struct{}

var _ ParseableType = DeducedParseableType{}

// IsValid always returns true for a DeducedParseableType.
func (p DeducedParseableType) IsValid() bool {
	return true
}

// FromYAML parses a yaml string into an object and deduces the type for
// that object.
func (p DeducedParseableType) FromYAML(object YAMLObject) (TypedValue, error) {
	v, err := value.FromYAML([]byte(object))
	if err != nil {
		return nil, err
	}
	return AsTypedDeduced(v), nil
}

// FromUnstructured converts a go interface to a TypedValue. It will return an
// error if the input object uses un-handled types.
func (p DeducedParseableType) FromUnstructured(in interface{}) (TypedValue, error) {
	v, err := value.FromUnstructured(in)
	if err != nil {
		return nil, err
	}
	return AsTypedDeduced(v), nil
}
