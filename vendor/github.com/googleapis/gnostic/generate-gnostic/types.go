// Copyright 2017 Google Inc. All Rights Reserved.
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

package main

import (
	"fmt"
	"strings"

	"github.com/googleapis/gnostic/jsonschema"
)

/// Type Modeling

// TypeRequest models types that we encounter during model-building that have no named schema.
type TypeRequest struct {
	Name         string             // name of type to be created
	PropertyName string             // name of a property that refers to this type
	Schema       *jsonschema.Schema // schema for type
	OneOfWrapper bool               // true if the type wraps "oneOfs"
}

// NewTypeRequest creates a TypeRequest.
func NewTypeRequest(name string, propertyName string, schema *jsonschema.Schema) *TypeRequest {
	return &TypeRequest{Name: name, PropertyName: propertyName, Schema: schema}
}

// TypeProperty models type properties, eg. fields.
type TypeProperty struct {
	Name             string   // name of property
	Type             string   // type for property (scalar or message type)
	StringEnumValues []string // possible values if this is an enumerated string type
	MapType          string   // if this property is for a map, the name of the mapped type
	Repeated         bool     // true if this property is repeated (an array)
	Pattern          string   // if the property is a pattern property, names must match this pattern.
	Implicit         bool     // true if this property is implied by a pattern or "additional properties" property
	Description      string   // if present, the "description" field in the schema
}

func (typeProperty *TypeProperty) description() string {
	result := ""
	if typeProperty.Description != "" {
		result += fmt.Sprintf("\t// %+s\n", typeProperty.Description)
	}
	if typeProperty.Repeated {
		result += fmt.Sprintf("\t%s %s repeated %s\n", typeProperty.Name, typeProperty.Type, typeProperty.Pattern)
	} else {
		result += fmt.Sprintf("\t%s %s %s \n", typeProperty.Name, typeProperty.Type, typeProperty.Pattern)
	}
	return result
}

// NewTypeProperty creates a TypeProperty
func NewTypeProperty() *TypeProperty {
	return &TypeProperty{}
}

// NewTypePropertyWithNameAndType creates a TypeProperty
func NewTypePropertyWithNameAndType(name string, typeName string) *TypeProperty {
	return &TypeProperty{Name: name, Type: typeName}
}

// NewTypePropertyWithNameTypeAndPattern creates a TypeProperty
func NewTypePropertyWithNameTypeAndPattern(name string, typeName string, pattern string) *TypeProperty {
	return &TypeProperty{Name: name, Type: typeName, Pattern: pattern}
}

// FieldName returns the message field name to use for a property.
func (typeProperty *TypeProperty) FieldName() string {
	propertyName := typeProperty.Name
	if propertyName == "$ref" {
		return "XRef"
	}
	return strings.Title(propertyName)
}

// TypeModel models types.
type TypeModel struct {
	Name          string          // type name
	Properties    []*TypeProperty // slice of properties
	Required      []string        // required property names
	OneOfWrapper  bool            // true if this type wraps "oneof" properties
	Open          bool            // open types can have keys outside the specified set
	OpenPatterns  []string        // patterns for properties that we allow
	IsStringArray bool            // ugly override
	IsItemArray   bool            // ugly override
	IsBlob        bool            // ugly override
	IsPair        bool            // type is a name-value pair used to support ordered maps
	PairValueType string          // type for pair values (valid if IsPair == true)
	Description   string          // if present, the "description" field in the schema
}

func (typeModel *TypeModel) addProperty(property *TypeProperty) {
	if typeModel.Properties == nil {
		typeModel.Properties = make([]*TypeProperty, 0)
	}
	typeModel.Properties = append(typeModel.Properties, property)
}

func (typeModel *TypeModel) description() string {
	result := ""
	if typeModel.Description != "" {
		result += fmt.Sprintf("// %+s\n", typeModel.Description)
	}
	var wrapperinfo string
	if typeModel.OneOfWrapper {
		wrapperinfo = " oneof wrapper"
	}
	result += fmt.Sprintf("%+s%s\n", typeModel.Name, wrapperinfo)
	for _, property := range typeModel.Properties {
		result += property.description()
	}
	return result
}

// NewTypeModel creates a TypeModel.
func NewTypeModel() *TypeModel {
	typeModel := &TypeModel{}
	typeModel.Properties = make([]*TypeProperty, 0)
	return typeModel
}
