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

	"gopkg.in/yaml.v2"

	"github.com/golang/glog"
	"github.com/googleapis/gnostic/OpenAPIv2"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// groupVersionKindExtensionKey is the key used to lookup the GroupVersionKind value
// for an object definition from the definition's "extensions" map.
const groupVersionKindExtensionKey = "x-kubernetes-group-version-kind"

// Integer is the name for integer types
const Integer = "integer"

// String is the name for string types
const String = "string"

// Bool is the name for boolean types
const Boolean = "boolean"

// Map is the name for map types
// types.go struct fields that are maps will have an open API type "object"
// types.go struct fields that are actual objects appearing as a struct
// in a types.go file will have no type defined
// and have a json pointer reference to the type definition
const Map = "object"

// Array is the name for array types
const Array = "array"

// Resources contains the object definitions for Kubernetes resource apis
// Fields are public for binary serialization (private fields don't get serialized)
type Resources struct {
	// GroupVersionKindToName maps GroupVersionKinds to Type names
	GroupVersionKindToName map[schema.GroupVersionKind]string
	// NameToDefinition maps Type names to TypeDefinitions
	NameToDefinition map[string]Kind
}

// LookupResource returns the Kind for the specified groupVersionKind
func (r Resources) LookupResource(groupVersionKind schema.GroupVersionKind) (Kind, bool) {
	name, found := r.GroupVersionKindToName[groupVersionKind]
	if !found {
		return Kind{}, false
	}
	def, found := r.NameToDefinition[name]
	if !found {
		return Kind{}, false
	}
	return def, true
}

// Kind defines a Kubernetes object Kind
type Kind struct {
	// Name is the lookup key given to this Kind by the open API spec.
	// May not contain any semantic meaning or relation to the API definition,
	// simply must be unique for each object definition in the Open API spec.
	// e.g. io.k8s.api.apps.v1beta1.Deployment
	Name string

	// IsResource is true if the Kind is a Resource (it has API endpoints)
	// e.g. Deployment is a Resource, DeploymentStatus is NOT a Resource
	IsResource bool

	// GroupVersionKind uniquely defines a resource type in the Kubernetes API
	// and is present for all resources.
	// Empty for non-resource Kinds (e.g. those without APIs).
	// e.g. "Group": "apps", "Version": "v1beta1", "Kind": "Deployment"
	GroupVersionKind schema.GroupVersionKind

	// Present only for definitions that represent primitive types with additional
	// semantic meaning beyond just string, integer, boolean - e.g.
	// Fields with a PrimitiveType should follow the validation of the primitive type.
	// io.k8s.apimachinery.pkg.apis.meta.v1.Time
	// io.k8s.apimachinery.pkg.util.intstr.IntOrString
	PrimitiveType string

	// Extensions are openapi extensions for the object definition.
	Extensions map[string]interface{}

	// Fields are the fields defined for this Kind
	Fields map[string]Type
}

// Type defines a field type and are expected to be one of:
// - IsKind
// - IsMap
// - IsArray
// - IsPrimitive
type Type struct {
	// Name is the name of the type
	TypeName string

	// IsKind is true if the definition represents a Kind
	IsKind bool
	// IsPrimitive is true if the definition represents a primitive type - e.g. string, boolean, integer
	IsPrimitive bool
	// IsArray is true if the definition represents an array type
	IsArray bool
	// IsMap is true if the definition represents a map type
	IsMap bool

	// ElementType will be specified for arrays and maps
	// if IsMap == true, then ElementType is the type of the value (key is always string)
	// if IsArray == true, then ElementType is the type of the element
	ElementType *Type

	// Extensions are extensions for this field and may contain
	// metadata from the types.go struct field tags.
	// e.g. contains patchStrategy, patchMergeKey, etc
	Extensions map[string]interface{}
}

func vendorExtensionToMap(e []*openapi_v2.NamedAny) map[string]interface{} {
	var values map[string]interface{}

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
		if values == nil {
			values = make(map[string]interface{})
		}
		values[na.GetName()] = value
	}

	return values
}

// NewOpenAPIData parses the resource definitions in openapi data by groupversionkind and name
func NewOpenAPIData(doc *openapi_v2.Document) (*Resources, error) {
	o := &Resources{
		GroupVersionKindToName: map[schema.GroupVersionKind]string{},
		NameToDefinition:       map[string]Kind{},
	}
	// Parse and index definitions by name
	for _, ns := range doc.GetDefinitions().GetAdditionalProperties() {
		definition := o.parseDefinition(ns.GetName(), ns.GetValue())
		o.NameToDefinition[ns.GetName()] = definition
		if len(definition.GroupVersionKind.Kind) > 0 {
			o.GroupVersionKindToName[definition.GroupVersionKind] = ns.GetName()
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
			for _, t := range o.getTypeNames(f) {
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

func (o *Resources) getTypeNames(elem Type) []string {
	t := []string{}
	if elem.IsKind {
		t = append(t, elem.TypeName)
	}
	if elem.ElementType != nil && elem.ElementType.IsKind {
		t = append(t, o.getTypeNames(*elem.ElementType)...)
	}
	return t
}

func (o *Resources) parseDefinition(name string, s *openapi_v2.Schema) Kind {
	gvk, err := o.getGroupVersionKind(s)
	value := Kind{
		Name:             name,
		GroupVersionKind: gvk,
		Extensions:       vendorExtensionToMap(s.GetVendorExtension()),
		Fields:           map[string]Type{},
	}
	if err != nil {
		glog.V(2).Info(err)
	}

	// Definition represents a primitive type - e.g.
	// io.k8s.apimachinery.pkg.util.intstr.IntOrString
	if o.isPrimitive(s) {
		value.PrimitiveType = o.getTypeNameForField(s)
	}
	for _, ns := range s.GetProperties().GetAdditionalProperties() {
		value.Fields[ns.GetName()] = o.parseField(ns.GetValue())
	}
	return value
}

func (o *Resources) parseField(s *openapi_v2.Schema) Type {
	def := Type{
		TypeName:    o.getTypeNameForField(s),
		IsPrimitive: o.isPrimitive(s),
		IsArray:     o.isArray(s),
		IsMap:       o.isMap(s),
		IsKind:      o.isDefinitionReference(s),
	}

	if elementType, arrayErr := o.getElementType(s); arrayErr == nil {
		d := o.parseField(elementType)
		def.ElementType = &d
	} else if valueType, mapErr := o.getValueType(s); mapErr == nil {
		d := o.parseField(valueType)
		def.ElementType = &d
	}

	def.Extensions = vendorExtensionToMap(s.GetVendorExtension())

	return def
}

// isArray returns true if s is an array type.
func (o *Resources) isArray(s *openapi_v2.Schema) bool {
	if len(s.GetProperties().GetAdditionalProperties()) > 0 {
		// Open API can have embedded type definitions, but Kubernetes doesn't generate these.
		// This should just be a sanity check against changing the format.
		return false
	}
	return o.getType(s) == Array
}

// isMap returns true if s is a map type.
func (o *Resources) isMap(s *openapi_v2.Schema) bool {
	if len(s.GetProperties().GetAdditionalProperties()) > 0 {
		// Open API can have embedded type definitions, but Kubernetes doesn't generate these.
		// This should just be a sanity check against changing the format.
		return false
	}
	return o.getType(s) == Map
}

// isPrimitive returns true if s is a primitive type
// Note: For object references that represent primitive types - e.g. IntOrString - this will
// be false, and the referenced Kind will have a non-empty "PrimitiveType".
func (o *Resources) isPrimitive(s *openapi_v2.Schema) bool {
	if len(s.GetProperties().GetAdditionalProperties()) > 0 {
		// Open API can have embedded type definitions, but Kubernetes doesn't generate these.
		// This should just be a sanity check against changing the format.
		return false
	}
	t := o.getType(s)
	if t == Integer || t == Boolean || t == String {
		return true
	}
	return false
}

func (*Resources) getType(s *openapi_v2.Schema) string {
	if len(s.GetType().GetValue()) != 1 {
		return ""
	}
	return strings.ToLower(s.GetType().GetValue()[0])
}

func (o *Resources) getTypeNameForField(s *openapi_v2.Schema) string {
	// Get the reference for complex types
	if o.isDefinitionReference(s) {
		return o.nameForDefinitionField(s)
	}
	// Recurse if type is array
	if o.isArray(s) {
		return fmt.Sprintf("%s array", o.getTypeNameForField(s.GetItems().GetSchema()[0]))
	}
	if o.isMap(s) {
		return fmt.Sprintf("%s map", o.getTypeNameForField(s.GetAdditionalProperties().GetSchema()))
	}

	// Get the value for primitive types
	if o.isPrimitive(s) {
		return fmt.Sprintf("%s", s.GetType().GetValue()[0])
	}
	return ""
}

// isDefinitionReference returns true s is a complex type that should have a Kind.
func (o *Resources) isDefinitionReference(s *openapi_v2.Schema) bool {
	if len(s.GetProperties().GetAdditionalProperties()) > 0 {
		// Open API can have embedded type definitions, but Kubernetes doesn't generate these.
		// This should just be a sanity check against changing the format.
		return false
	}
	if len(s.GetType().GetValue()) > 0 {
		// Definition references won't have a type
		return false
	}

	p := s.GetXRef()
	return len(p) > 0 && strings.HasPrefix(p, "#/definitions/")
}

// getElementType returns the type of an element for arrays
// returns an error if s is not an array.
func (o *Resources) getElementType(s *openapi_v2.Schema) (*openapi_v2.Schema, error) {
	if !o.isArray(s) {
		return &openapi_v2.Schema{}, fmt.Errorf("%v is not an array type", o.getTypeNameForField(s))
	}
	return s.GetItems().GetSchema()[0], nil
}

// getValueType returns the type of an element for maps
// returns an error if s is not a map.
func (o *Resources) getValueType(s *openapi_v2.Schema) (*openapi_v2.Schema, error) {
	if !o.isMap(s) {
		return &openapi_v2.Schema{}, fmt.Errorf("%v is not an map type", o.getTypeNameForField(s))
	}
	return s.GetAdditionalProperties().GetSchema(), nil
}

// nameForDefinitionField returns the definition name for the schema (field) if it is a complex type
func (o *Resources) nameForDefinitionField(s *openapi_v2.Schema) string {
	p := s.GetXRef()
	if len(p) == 0 {
		return ""
	}

	// Strip the "definitions/" pieces of the reference
	return strings.Replace(p, "#/definitions/", "", -1)
}

// getGroupVersionKind implements OpenAPIData
// getGVK parses the gropuversionkind for a resource definition from the x-kubernetes
// extensions
// map[x-kubernetes-group-version-kind:[map[Group:authentication.k8s.io Version:v1 Kind:TokenReview]]]
func (o *Resources) getGroupVersionKind(s *openapi_v2.Schema) (schema.GroupVersionKind, error) {
	empty := schema.GroupVersionKind{}

	extensionMap := vendorExtensionToMap(s.GetVendorExtension())
	// Get the extensions
	extList, f := extensionMap[groupVersionKindExtensionKey]
	if !f {
		return empty, fmt.Errorf("No %s extension present in %v", groupVersionKindExtensionKey, extensionMap)
	}

	// Expect a empty of a list with 1 element
	extListCasted, ok := extList.([]interface{})
	if !ok {
		return empty, fmt.Errorf("%s extension has unexpected type %T in %s", groupVersionKindExtensionKey, extListCasted, extensionMap)
	}
	if len(extListCasted) == 0 {
		return empty, fmt.Errorf("No Group Version Kind found in %v", extListCasted)
	}
	if len(extListCasted) != 1 {
		return empty, fmt.Errorf("Multiple Group Version gvkToName found in %v", extListCasted)
	}
	gvk := extListCasted[0]

	// Expect a empty of a map with 3 entries
	gvkMap, ok := gvk.(map[interface{}]interface{})
	if !ok {
		return empty, fmt.Errorf("%s extension has unexpected type %T in %s", groupVersionKindExtensionKey, gvk, extList)
	}
	group, ok := gvkMap["group"].(string)
	if !ok {
		return empty, fmt.Errorf("%s extension missing Group: %v", groupVersionKindExtensionKey, gvkMap)
	}
	version, ok := gvkMap["version"].(string)
	if !ok {
		return empty, fmt.Errorf("%s extension missing Version: %v", groupVersionKindExtensionKey, gvkMap)
	}
	kind, ok := gvkMap["kind"].(string)
	if !ok {
		return empty, fmt.Errorf("%s extension missing Kind: %v", groupVersionKindExtensionKey, gvkMap)
	}

	return schema.GroupVersionKind{
		Group:   group,
		Version: version,
		Kind:    kind,
	}, nil
}
