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

	"github.com/go-openapi/spec"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

const gvkOpenAPIKey = "x-kubernetes-group-version-kind"

// Integer is the name for integer types
const Integer = "integer"

// String is the name for string types
const String = "string"

// Bool is the name for boolean types
const Boolean = "boolean"

// Resources contains the object definitions for Kubernetes resource apis
type Resources struct {
	// GroupVersionKindToName maps GroupVersionKinds to TypeDefinition names
	GroupVersionKindToName map[schema.GroupVersionKind]string
	// NameToDefinition maps TypeDefinition names to TypeDefinitions
	NameToDefinition map[string]KindDefinition
}

// KindDefinition defines a Kubernetes object Kind
type KindDefinition struct {
	// Name is the name of the type
	Name string

	// IsResource is true if the Kind is a Resource (has API endpoints)
	IsResource bool

	// GroupVersionKind is the group version kind of a resource.
	// Empty for non-resource Kinds (e.g. those without APIs).
	GroupVersionKind schema.GroupVersionKind

	// Present for definitions that are primitives - e.g.
	// io.k8s.apimachinery.pkg.apis.meta.v1.Time
	// io.k8s.apimachinery.pkg.util.intstr.IntOrString
	PrimitiveType string

	// Extensions are the openapi extensions for the object definition.
	Extensions spec.Extensions

	// Fields is the list of fields for definitions that are Kinds
	Fields map[string]FieldDefinition
}

// TypeDefinition defines a field definition
type TypeDefinition struct {
	// Name is the name of the type
	TypeName string

	// IsKind is true if the definition represents a Kind
	IsKind bool
	// isPrimitive is true if the definition represents a primitive type
	IsPrimitive bool
	// isArray is true if the definition represents an array type
	IsArray bool
	// isArray is true if the definition represents an array type
	IsMap bool

	ElementType *TypeDefinition
}

// FieldDefinition defines a field
type FieldDefinition struct {
	TypeDefinition

	// Extensions are extensions for this field
	Extensions spec.Extensions
}

// newOpenAPIData parses the resource definitions in openapi data by groupversionkind and name
func newOpenAPIData(s *spec.Swagger) (*Resources, error) {
	o := &Resources{
		GroupVersionKindToName: map[schema.GroupVersionKind]string{},
		NameToDefinition:       map[string]KindDefinition{},
	}
	// Parse and index definitions by name
	for name, d := range s.Definitions {
		definition := o.parseDefinition(name, d)
		o.NameToDefinition[name] = definition
		if len(definition.GroupVersionKind.Kind) > 0 {
			o.GroupVersionKindToName[definition.GroupVersionKind] = name
		}
	}

	if err := o.validate(); err != nil {
		return nil, err
	}

	return o, nil
}

// validate makes sure the definition for each field type is found in the map
func (o *Resources) validate() error {
	types := sets.String{}
	for _, d := range o.NameToDefinition {
		for _, f := range d.Fields {
			for _, t := range o.getTypeNames(f.TypeDefinition) {
				types.Insert(t)
			}
		}
	}
	for _, n := range types.List() {
		_, found := o.NameToDefinition[n]
		if !found {
			return fmt.Errorf("Unable to find definition for field of type %v", n)
		}
	}
	return nil
}

func (o *Resources) getTypeNames(elem TypeDefinition) []string {
	t := []string{}
	if elem.IsKind {
		t = append(t, elem.TypeName)
	}
	if elem.ElementType != nil && elem.ElementType.IsKind {
		t = append(t, o.getTypeNames(*elem.ElementType)...)
	}
	return t
}

func (o *Resources) parseDefinition(name string, s spec.Schema) KindDefinition {
	gvk, err := o.getGroupVersionKind(s)
	value := KindDefinition{
		Name:             name,
		GroupVersionKind: gvk,
		Extensions:       s.Extensions,
		Fields:           map[string]FieldDefinition{},
	}
	if err != nil {
		glog.Warning(err)
	}

	// Definition represents a primitive type - e.g.
	// io.k8s.apimachinery.pkg.util.intstr.IntOrString
	if o.isPrimitive(s) {
		value.PrimitiveType = o.getTypeNameForField(s)
	}
	for fieldname, property := range s.Properties {
		value.Fields[fieldname] = o.parseField(fieldname, property)
	}
	return value
}

func (o *Resources) buildElementFromSchema(s spec.Schema) TypeDefinition {
	def := TypeDefinition{
		TypeName:    o.getTypeNameForField(s),
		IsPrimitive: o.isPrimitive(s),
		IsArray:     o.isArray(s),
		IsMap:       o.isMap(s),
		IsKind:      o.isDefinitionReference(s),
	}

	if elementType, arrayErr := o.getElementType(s); arrayErr == nil {
		d := o.buildElementFromSchema(elementType)
		def.ElementType = &d
	} else if valueType, mapErr := o.getValueType(s); mapErr == nil {
		d := o.buildElementFromSchema(valueType)
		def.ElementType = &d
	}

	return def
}

func (o *Resources) parseField(name string, s spec.Schema) FieldDefinition {
	fieldDef := FieldDefinition{
		Extensions:     s.Extensions,
		TypeDefinition: o.buildElementFromSchema(s),
	}
	return fieldDef
}

// isArray returns true if s is an array type.
func (o *Resources) isArray(s spec.Schema) bool {
	if len(s.Properties) > 0 {
		// Open API can have embedded type definitions, but Kubernetes doesn't generate these.
		// This should just be a sanity check against changing the format.
		return false
	}
	return len(s.Type) > 0 && s.Type[0] == "array"
}

// isMap returns true if s is a map type.
func (o *Resources) isMap(s spec.Schema) bool {
	if len(s.Properties) > 0 {
		// Open API can have embedded type definitions, but Kubernetes doesn't generate these.
		// This should just be a sanity check against changing the format.
		return false
	}
	return len(s.Type) > 0 && s.Type[0] == "object"
}

// isPrimitive returns true if s is a primitive type
// Note: For object references that represent primitive types - e.g. IntOrString - this will
// be false, and the referenced Kind will have a non-empty "PrimitiveType".
func (o *Resources) isPrimitive(s spec.Schema) bool {
	if len(s.Properties) > 0 {
		// Open API can have embedded type definitions, but Kubernetes doesn't generate these.
		// This should just be a sanity check against changing the format.
		return false
	}
	if len(s.Type) == 1 {
		switch s.Type[0] {
		case "integer":
			return true
		case "boolean":
			return true
		case "string":
			return true
		default:
			return false
		}
	}
	return false
}

func (o *Resources) getTypeNameForField(s spec.Schema) string {
	// Get the reference for complex types
	if o.isDefinitionReference(s) {
		return o.nameForDefinitionField(s)
	}
	// Recurse if type is array
	if o.isArray(s) {
		return fmt.Sprintf("%s array", o.getTypeNameForField(*s.Items.Schema))
	}
	if o.isMap(s) {
		return fmt.Sprintf("%s map", o.getTypeNameForField(*s.AdditionalProperties.Schema))
	}

	// Get the value for primitive types
	if o.isPrimitive(s) {
		return fmt.Sprintf("%s", s.Type[0])
	}
	return ""
}

// isDefinitionReference returns true s is a complex type that should have a KindDefinition.
func (o *Resources) isDefinitionReference(s spec.Schema) bool {
	if len(s.Properties) > 0 {
		// Open API can have embedded type definitions, but Kubernetes doesn't generate these.
		// This should just be a sanity check against changing the format.
		return false
	}
	if len(s.Type) > 0 {
		// Definition references won't have a type
		return false
	}

	p := s.SchemaProps.Ref.GetPointer().String()
	return len(p) > 0 && strings.HasPrefix(p, "/definitions/")
}

// getElementType returns the type of an element for arrays and maps
func (o *Resources) getElementType(s spec.Schema) (spec.Schema, error) {
	if !o.isArray(s) {
		return spec.Schema{}, fmt.Errorf("%v is not an array type", s.Type)
	}
	return *s.Items.Schema, nil
}

func (o *Resources) getValueType(s spec.Schema) (spec.Schema, error) {
	if !o.isMap(s) {
		return spec.Schema{}, fmt.Errorf("%v is not an map type", s.Type)
	}
	return *s.AdditionalProperties.Schema, nil
}

// nameForDefinitionField returns the definition name for the schema (field) if it is a complex type
func (o *Resources) nameForDefinitionField(s spec.Schema) string {
	p := s.SchemaProps.Ref.GetPointer().String()
	if len(p) == 0 {
		return ""
	}

	// Strip the "definitions/" pieces of the reference
	return strings.Replace(p, "/definitions/", "", -1)
}

// getGroupVersionKind implements openAPIData
// getGVK parses the gropuversionkind for a resource definition from the x-kubernetes
// extensions
// Expected format for s.Extensions: map[string][]map[string]string
// map[x-kubernetes-group-version-kind:[map[Group:authentication.k8s.io Version:v1 Kind:TokenReview]]]
func (o *Resources) getGroupVersionKind(s spec.Schema) (schema.GroupVersionKind, error) {
	empty := schema.GroupVersionKind{}

	// Get the extensions
	extList, f := s.Extensions[gvkOpenAPIKey]
	if !f {
		return empty, fmt.Errorf("No %s extension present in %v", gvkOpenAPIKey, s.Extensions)
	}

	// Expect a empty of a list with 1 element
	extListCasted, ok := extList.([]interface{})
	if !ok {
		return empty, fmt.Errorf("%s extension has unexpected type %T in %s", gvkOpenAPIKey, extListCasted, s.Extensions)
	}
	if len(extListCasted) == 0 {
		return empty, fmt.Errorf("No Group Version Kind found in %v", extListCasted)
	}
	if len(extListCasted) != 1 {
		return empty, fmt.Errorf("Multiple Group Version gvkToName found in %v", extListCasted)
	}
	gvk := extListCasted[0]

	// Expect a empty of a map with 3 entries
	gvkMap, ok := gvk.(map[string]interface{})
	if !ok {
		return empty, fmt.Errorf("%s extension has unexpected type %T in %s", gvkOpenAPIKey, gvk, s.Extensions)
	}
	group, ok := gvkMap["Group"].(string)
	if !ok {
		return empty, fmt.Errorf("%s extension missing Group: %v", gvkOpenAPIKey, gvkMap)
	}
	version, ok := gvkMap["Version"].(string)
	if !ok {
		return empty, fmt.Errorf("%s extension missing Version: %v", gvkOpenAPIKey, gvkMap)
	}
	kind, ok := gvkMap["Kind"].(string)
	if !ok {
		return empty, fmt.Errorf("%s extension missing Kind: %v", gvkOpenAPIKey, gvkMap)
	}

	return schema.GroupVersionKind{
		Group:   group,
		Version: version,
		Kind:    kind,
	}, nil
}
