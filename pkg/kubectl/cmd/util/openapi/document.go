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
	"fmt"
	"strings"

	openapi_v2 "github.com/googleapis/gnostic/OpenAPIv2"
	yaml "gopkg.in/yaml.v2"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func newSchemaError(path *Path, format string, a ...interface{}) error {
	err := fmt.Sprintf(format, a...)
	if path.Len() == 0 {
		return fmt.Errorf("SchemaError: %v", err)
	}
	return fmt.Errorf("SchemaError(%v): %v", path, err)
}

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

	// gvk extension must be a list of 1 element.
	gvkList, ok := gvkExtension.([]interface{})
	if !ok {
		return schema.GroupVersionKind{}
	}
	if len(gvkList) != 1 {
		return schema.GroupVersionKind{}

	}
	gvk := gvkList[0]

	// gvk extension list must be a map with group, version, and
	// kind fields
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
		gvk := parseGroupVersionKind(namedSchema.GetValue())
		if len(gvk.Kind) > 0 {
			definitions.resources[gvk] = namedSchema.GetName()
		}
	}

	return &definitions, nil
}

// We believe the schema is a reference, verify that and returns a new
// Schema
func (d *Definitions) parseReference(s *openapi_v2.Schema, path *Path) (Schema, error) {
	if len(s.GetProperties().GetAdditionalProperties()) > 0 {
		return nil, newSchemaError(path, "unallowed embedded type definition")
	}
	if len(s.GetType().GetValue()) > 0 {
		return nil, newSchemaError(path, "definition reference can't have a type")
	}

	if !strings.HasPrefix(s.GetXRef(), "#/definitions/") {
		return nil, newSchemaError(path, "unallowed reference to non-definition %q", s.GetXRef())
	}
	reference := strings.TrimPrefix(s.GetXRef(), "#/definitions/")
	if _, ok := d.models[reference]; !ok {
		return nil, newSchemaError(path, "unknown model in reference: %q", reference)
	}
	return &Reference{
		Reference:   reference,
		definitions: d,
	}, nil
}

func (d *Definitions) parseBaseSchema(s *openapi_v2.Schema, path *Path) BaseSchema {
	return BaseSchema{
		Description: s.GetDescription(),
		Extensions:  vendorExtensionToMap(s.GetVendorExtension()),
		Path:        *path,
	}
}

// We believe the schema is a map, verify and return a new schema
func (d *Definitions) parseMap(s *openapi_v2.Schema, path *Path) (Schema, error) {
	if len(s.GetType().GetValue()) != 0 && s.GetType().GetValue()[0] != object {
		return nil, newSchemaError(path, "invalid object type")
	}
	if s.GetAdditionalProperties().GetSchema() == nil {
		return nil, newSchemaError(path, "invalid object doesn't have additional properties")
	}
	sub, err := d.ParseSchema(s.GetAdditionalProperties().GetSchema(), path)
	if err != nil {
		return nil, err
	}
	return &Map{
		BaseSchema: d.parseBaseSchema(s, path),
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
	case String:
	case Number:
	case Integer:
	case Boolean:
	case "": // Some models are completely empty, and can be safely ignored.
		// Do nothing
	default:
		return nil, newSchemaError(path, "Unknown primitive type: %q", t)
	}
	return &Primitive{
		BaseSchema: d.parseBaseSchema(s, path),
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
		return nil, newSchemaError(path, "array should have exactly one sub-item")
	}
	sub, err := d.ParseSchema(s.GetItems().GetSchema()[0], path)
	if err != nil {
		return nil, err
	}
	return &Array{
		BaseSchema: d.parseBaseSchema(s, path),
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

	for _, namedSchema := range s.GetProperties().GetAdditionalProperties() {
		var err error
		path := path.FieldPath(namedSchema.GetName())
		fields[namedSchema.GetName()], err = d.ParseSchema(namedSchema.GetValue(), &path)
		if err != nil {
			return nil, err
		}
	}

	return &Kind{
		BaseSchema:     d.parseBaseSchema(s, path),
		RequiredFields: s.GetRequired(),
		Fields:         fields,
	}, nil
}

// ParseSchema creates a walkable Schema from an openapi schema. While
// this function is public, it doesn't leak through the interface.
func (d *Definitions) ParseSchema(s *openapi_v2.Schema, path *Path) (Schema, error) {
	if len(s.GetType().GetValue()) == 1 {
		t := s.GetType().GetValue()[0]
		switch t {
		case object:
			return d.parseMap(s, path)
		case array:
			return d.parseArray(s, path)
		}

	}
	if s.GetXRef() != "" {
		return d.parseReference(s, path)
	}
	if s.GetProperties() != nil {
		return d.parseKind(s, path)
	}
	return d.parsePrimitive(s, path)
}

// LookupResource is public through the interface of Resources. It
// returns a visitable schema from the given group-version-kind.
func (d *Definitions) LookupResource(gvk schema.GroupVersionKind) Schema {
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

func (*Reference) GetPath() *Path {
	// Reference never has a path, because it can be referenced from
	// multiple locations.
	return &Path{}
}

func (r *Reference) GetName() string {
	return r.Reference
}
