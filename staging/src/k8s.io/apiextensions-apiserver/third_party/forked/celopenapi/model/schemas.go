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


// SchemaDeclTypes constructs a top-down set of DeclType instances whose name is derived from the root
// type name provided on the call, if not set to a custom type.
func SchemaDeclTypes(s *schema.Structural, maybeRootType string) (*DeclType, map[string]*DeclType) {
	root := SchemaDeclType(s).MaybeAssignTypeName(maybeRootType)
	types := FieldTypeMap(maybeRootType, root)
	return root, types
}

// SchemaDeclType returns the cel type name associated with the schema element.
func SchemaDeclType(s *schema.Structural) *DeclType {
	if s == nil {
		return nil
	}
	if s.XIntOrString {
		// schemas using this extension are not required to have a type, so they must be handled before type lookup
		return intOrStringType
	}
	declType, found := openAPISchemaTypes[s.Type]
	if !found {
		return nil
	}

	// We ignore XPreserveUnknownFields since we don't support validation rules on
	// data that we don't have schema information for.

	if s.XEmbeddedResource {
		// 'apiVersion', 'kind', 'metadata.name' and 'metadata.generateName' are always accessible
		// to validation rules since this part of the schema is well known and validated when CRDs
		// are created and updated.
		s = WithTypeAndObjectMeta(s)
	}

	switch declType.TypeName() {
	case ListType.TypeName():
		return NewListType(SchemaDeclType(s.Items))
	case MapType.TypeName():
		if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
			return NewMapType(StringType, SchemaDeclType(s.AdditionalProperties.Structural))
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
			if fieldType := SchemaDeclType(&prop); fieldType != nil {
				fields[Escape(name)] = &DeclField{
					Name:         Escape(name),
					Required:     required[name],
					Type:         fieldType,
					defaultValue: prop.Default.Object,
					enumValues:   enumValues, // Enum values are represented as strings in CEL
				}
			}
		}
		return NewObjectType("object", fields)
	case StringType.TypeName():
		if s.ValueValidation != nil {
			switch s.ValueValidation.Format {
			case "byte":
				return StringType // OpenAPIv3 byte format represents base64 encoded string
			case "binary":
				return BytesType
			case "duration":
				return DurationType
			case "date", "date-time":
				return TimestampType
			}
		}
	}
	return declType
}

var (
	openAPISchemaTypes = map[string]*DeclType{
		"boolean":         BoolType,
		"number":          DoubleType,
		"integer":         IntType,
		"null":            NullType,
		"string":          StringType,
		"date":            DateType,
		"array":           ListType,
		"object":          MapType,
	}

	// intOrStringType represents the x-kubernetes-int-or-string union type in CEL expressions.
	// In CEL, the type is represented as an object where either the srtVal
	// or intVal field is set. In CEL, this allows for typesafe expressions like:
	//
	// require that the string representation be a percentage:
	//  `has(intOrStringField.strVal) && intOrStringField.strVal.matches(r'(\d+(\.\d+)?%)')`
	// validate requirements on both the int and string representation:
	//  `has(intOrStringField.intVal) ? intOrStringField.intVal < 5 : double(intOrStringField.strVal.replace('%', '')) < 0.5
	//
	intOrStringType = NewObjectType("intOrString", map[string]*DeclField{
		"strVal": {Name: "strVal", Type: StringType},
		"intVal": {Name: "intVal", Type: IntType},
	})
)

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