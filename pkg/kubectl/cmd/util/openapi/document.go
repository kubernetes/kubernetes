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

package openapi

import (
	"errors"
	"fmt"
	"strings"

	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	yaml "gopkg.in/yaml.v2"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// groupVersionKindExtensionKey is the key used to lookup the
// GroupVersionKind value for an object definition from the
// definition's "extensions" map.
const groupVersionKindExtensionKey = "x-kubernetes-group-version-kind"

func vendorExtensionToMap(e []*openapi_v2.NamedAny) map[string]interface{} {
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

// Get and parse GroupVersionKind from the extension. Returns empty if it doesn't have one.
func parseGroupVersionKind(s *openapi_v2.Schema) schema.GroupVersionKind {
	extensionMap := vendorExtensionToMap(s.GetVendorExtension())

	// Get the extensions
	gvkExtension, ok := extensionMap[groupVersionKindExtensionKey]
	if !ok {
		return schema.GroupVersionKind{}
	}

	// Expect a empty of a list with 1 element
	gvkList, ok := gvkExtension.([]interface{})
	if !ok {
		return schema.GroupVersionKind{}
	}
	if len(gvkList) != 1 {
		return schema.GroupVersionKind{}

	}
	gvk := gvkList[0]

	// Expect a empty of a map with 3 entries
	gvkMap, ok := gvk.(map[interface{}]interface{})
	if !ok {
		return schema.GroupVersionKind{}
	}
	group, ok := gvkMap["group"].(string)
	if !ok {
		return schema.GroupVersionKind{}
	}
	version, ok := gvkMap["version"].(string)
	if !ok {
		return schema.GroupVersionKind{}
	}
	kind, ok := gvkMap["kind"].(string)
	if !ok {
		return schema.GroupVersionKind{}
	}

	return schema.GroupVersionKind{
		Group:   group,
		Version: version,
		Kind:    kind,
	}
}

// Definitions is an implementation of `Resources`. It looks for
// resources in an openapi Schema.
type Definitions struct {
	models    map[string]Schema
	resources map[schema.GroupVersionKind]string
}

var _ Resources = &Definitions{}

// NewOpenAPIData creates a new `Resources` out of the openapi document.
func NewOpenAPIData(doc *openapi_v2.Document) (Resources, error) {
	definitions := Definitions{
		models:    map[string]Schema{},
		resources: map[schema.GroupVersionKind]string{},
	}

	// Collect the list of models first.
	for _, namedSchema := range doc.GetDefinitions().GetAdditionalProperties() {
		definitions.models[namedSchema.GetName()] = nil
	}

	// Now, parse each model. We can make sure reference exists.
	for _, namedSchema := range doc.GetDefinitions().GetAdditionalProperties() {
		schema, err := definitions.ParseSchema(namedSchema.GetValue())
		if err != nil {
			return nil, err
		}
		definitions.models[namedSchema.GetName()] = schema
		gvk := parseGroupVersionKind(namedSchema.GetValue())
		if len(gvk.Kind) > 0 {
			definitions.resources[gvk] = namedSchema.GetName()
		}
	}

	return definitions, nil
}

// We believe the schema is a reference, verify that and returns a new
// Schema
func (d *Definitions) parseReference(s *openapi_v2.Schema) (Schema, error) {
	if len(s.GetProperties().GetAdditionalProperties()) > 0 {
		return nil, errors.New("unallowed embedded type definition")
	}
	if len(s.GetType().GetValue()) > 0 {
		return nil, errors.New("definition reference can't have a type")
	}

	if !strings.HasPrefix(s.GetXRef(), "#/definitions/") {
		return nil, fmt.Errorf("unallowed reference to non-definition (%s)", s.GetXRef())
	}
	reference := strings.TrimPrefix(s.GetXRef(), "#/definitions/")
	if _, ok := d.models[reference]; !ok {
		return nil, fmt.Errorf("unknown model in reference: %s", reference)
	}
	return &Reference{
		Reference:   reference,
		definitions: d,
	}, nil
}

func (d *Definitions) parseBaseSchema(s *openapi_v2.Schema) BaseSchema {
	return BaseSchema{
		Description: s.GetDescription(),
		Extensions:  vendorExtensionToMap(s.GetVendorExtension()),
	}
}

// We believe the schema is a simpleobject, verify and return a new schema
func (d *Definitions) parseSimpleObject(s *openapi_v2.Schema) (Schema, error) {
	if len(s.GetType().GetValue()) != 0 && s.GetType().GetValue()[0] != "object" {
		return nil, errors.New("invalid object type")
	}
	if s.GetAdditionalProperties().GetSchema() == nil {
		fmt.Println(s)
		return nil, errors.New("invalid object doesn't have additional properties")
	}
	sub, err := d.ParseSchema(s.GetAdditionalProperties().GetSchema())
	if err != nil {
		return nil, err
	}
	return &SimpleObject{
		BaseSchema: d.parseBaseSchema(s),
		SubType:    sub,
	}, nil
}

func (d *Definitions) parsePrimitive(s *openapi_v2.Schema) (Schema, error) {
	var t string
	if len(s.GetType().GetValue()) > 1 {
		return nil, errors.New("primitive can't have more than 1 type")
	}
	if len(s.GetType().GetValue()) == 1 {
		t = s.GetType().GetValue()[0]
	}
	return &Primitive{
		BaseSchema: d.parseBaseSchema(s),
		Type:       t,
		Format:     s.GetFormat(),
	}, nil
}

func (d Definitions) parseArray(s *openapi_v2.Schema) (Schema, error) {
	if len(s.GetType().GetValue()) != 1 {
		return nil, errors.New("array should have exactly one type")
	}
	if s.GetType().GetValue()[0] != "array" {
		return nil, errors.New(`array should have type "array"`)
	}
	if len(s.GetItems().GetSchema()) != 1 {
		return nil, errors.New("array should have exactly one sub-item")
	}
	sub, err := d.ParseSchema(s.GetItems().GetSchema()[0])
	if err != nil {
		return nil, err
	}
	return &Array{
		BaseSchema: d.parseBaseSchema(s),
		SubType:    sub,
	}, nil
}

func (d *Definitions) parsePropertiesMap(s *openapi_v2.Schema) (Schema, error) {
	if len(s.GetType().GetValue()) != 0 && s.GetType().GetValue()[0] != "object" {
		return nil, errors.New("invalid object type")
	}
	if s.GetProperties() == nil {
		return nil, errors.New("object doesn't have properties")
	}

	fields := map[string]Schema{}

	for _, namedSchema := range s.GetProperties().GetAdditionalProperties() {
		var err error
		fields[namedSchema.GetName()], err = d.ParseSchema(namedSchema.GetValue())
		if err != nil {
			return nil, err
		}
	}

	return &PropertiesMap{
		BaseSchema:     d.parseBaseSchema(s),
		RequiredFields: s.GetRequired(),
		Fields:         fields,
	}, nil
}

// ParseSchema creates a walkable Schema from an openapi schema. While
// this function is public, it doesn't leak through the interface.
func (d *Definitions) ParseSchema(s *openapi_v2.Schema) (Schema, error) {
	if s.GetXRef() != "" {
		return d.parseReference(s)
	}
	if s.GetAdditionalProperties() != nil {
		return d.parseSimpleObject(s)
	}
	if s.GetProperties() != nil {
		return d.parsePropertiesMap(s)
	}
	if s.GetItems() != nil {
		return d.parseArray(s)
	}
	return d.parsePrimitive(s)
}

// LookupResource is public through the interface of Resources. It
// returns a visitable schema from the given group-version-kind.
func (d Definitions) LookupResource(gvk schema.GroupVersionKind) Schema {
	modelName, found := d.resources[gvk]
	if !found {
		return nil
	}
	model, found := d.models[modelName]
	if !found {
		return nil
	}
	return model
}

// ListModels returns the list of all known models
func (d Definitions) ListGroupVersionKinds() []schema.GroupVersionKind {
	kinds := []schema.GroupVersionKind{}
	for kind := range d.resources {
		kinds = append(kinds, kind)
	}
	return kinds
}

// SchemaReference doesn't match a specific type. It's mostly a
// pass-through type.
type Reference struct {
	Reference string

	definitions *Definitions
}

var _ Schema = &Reference{}

func (r *Reference) GetSubSchema() Schema {
	return r.definitions.models[r.Reference]
}

func (r *Reference) Accept(s SchemaVisitor) {
	r.GetSubSchema().Accept(s)
}

func (r *Reference) GetDescription() string {
	return r.GetSubSchema().GetDescription()
}

func (r *Reference) GetExtensions() map[string]interface{} {
	return r.GetSubSchema().GetExtensions()
}
