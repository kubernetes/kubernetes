// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

// SchemaDeclType converts the structural schema to a CEL declaration, or returns nil if the
// the structural schema should not be exposed in CEL expressions.
// Set isResourceRoot to true for the root of a custom resource or embedded resource.
//
// Schemas with XPreserveUnknownFields not exposed unless they are objects. Array and "maps" schemas
// are not exposed if their items or additionalProperties schemas are not exposed. Object Properties are not exposed
// if their schema is not exposed.
//
// The CEL declaration for objects with XPreserveUnknownFields does not expose unknown fields.
func SchemaDeclType(s *schema.Structural, isResourceRoot bool) *DeclType {
	if s == nil {
		return nil
	}
	if s.XIntOrString {
		// schemas using XIntOrString are not required to have a type.

		// intOrStringType represents the x-kubernetes-int-or-string union type in CEL expressions.
		// In CEL, the type is represented as dynamic value, which can be thought of as a union type of all types.
		// All type checking for XIntOrString is deferred to runtime, so all access to values of this type must
		// be guarded with a type check, e.g.:
		//
		// To require that the string representation be a percentage:
		//  `type(intOrStringField) == string && intOrStringField.matches(r'(\d+(\.\d+)?%)')`
		// To validate requirements on both the int and string representation:
		//  `type(intOrStringField) == int ? intOrStringField < 5 : double(intOrStringField.replace('%', '')) < 0.5
		//
		return DynType
	}

	// We ignore XPreserveUnknownFields since we don't support validation rules on
	// data that we don't have schema information for.

	if isResourceRoot {
		// 'apiVersion', 'kind', 'metadata.name' and 'metadata.generateName' are always accessible to validator rules
		// at the root of resources, even if not specified in the schema.
		// This includes the root of a custom resource and the root of XEmbeddedResource objects.
		s = WithTypeAndObjectMeta(s)
	}

	switch s.Type {
	case "array":
		if s.Items != nil {
			itemsType := SchemaDeclType(s.Items, s.Items.XEmbeddedResource)
			if itemsType != nil {
				return NewListType(itemsType)
			}
		}
		return nil
	case "object":
		if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
			propsType := SchemaDeclType(s.AdditionalProperties.Structural, s.AdditionalProperties.Structural.XEmbeddedResource)
			if propsType != nil {
				return NewMapType(StringType, propsType)
			}
			return nil
		}
		fields := make(map[string]*DeclField, len(s.Properties))

		required := map[string]bool{}
		if s.ValueValidation != nil {
			for _, f := range s.ValueValidation.Required {
				required[f] = true
			}
		}
		for name, prop := range s.Properties {
			var enumValues []interface{}
			if prop.ValueValidation != nil {
				for _, e := range prop.ValueValidation.Enum {
					enumValues = append(enumValues, e.Object)
				}
			}
			if fieldType := SchemaDeclType(&prop, prop.XEmbeddedResource); fieldType != nil {
				if propName, ok := Escape(name); ok {
					fields[propName] = &DeclField{
						Name:         propName,
						Required:     required[name],
						Type:         fieldType,
						defaultValue: prop.Default.Object,
						enumValues:   enumValues, // Enum values are represented as strings in CEL
					}
				}
			}
		}
		return NewObjectType("object", fields)
	case "string":
		if s.ValueValidation != nil {
			switch s.ValueValidation.Format {
			case "byte":
				return BytesType
			case "duration":
				return DurationType
			case "date", "date-time":
				return TimestampType
			}
		}
		return StringType
	case "boolean":
		return BoolType
	case "number":
		return DoubleType
	case "integer":
		return IntType
	}
	return nil
}

// WithTypeAndObjectMeta ensures the kind, apiVersion and
// metadata.name and metadata.generateName properties are specified, making a shallow copy of the provided schema if needed.
func WithTypeAndObjectMeta(s *schema.Structural) *schema.Structural {
	if s.Properties != nil &&
		s.Properties["kind"].Type == "string" &&
		s.Properties["apiVersion"].Type == "string" &&
		s.Properties["metadata"].Type == "object" &&
		s.Properties["metadata"].Properties != nil &&
		s.Properties["metadata"].Properties["name"].Type == "string" &&
		s.Properties["metadata"].Properties["generateName"].Type == "string" {
		return s
	}
	result := &schema.Structural{
		Generic: s.Generic,
		Extensions: s.Extensions,
		ValueValidation: s.ValueValidation,
	}
	props := make(map[string]schema.Structural, len(s.Properties))
	for k, prop := range s.Properties {
		props[k] = prop
	}
	stringType := schema.Structural{Generic: schema.Generic{Type: "string"}}
	props["kind"] = stringType
	props["apiVersion"] = stringType
	props["metadata"] = schema.Structural{
		Generic: schema.Generic{Type: "object"},
		Properties: map[string]schema.Structural{
			"name": stringType,
			"generateName": stringType,
		},
	}
	result.Properties = props

	return result
}