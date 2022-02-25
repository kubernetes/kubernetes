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
	"time"

	"github.com/google/cel-go/checker/decls"
	"github.com/google/cel-go/common/types"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

const (
	// the largest request that will be accepted is 3MB
	// TODO(DangerOnTheRanger): find out where this is originally declared so we don't have two constants
	maxRequestSizeBytes = 3000000
	// chosen as the numbers of digits of the largest 64-bit integer
	// we add 4 to account for a unit (like ms) plus the quotation marks that will be used
	maxDurationSizeJSON = 23
	// OpenAPI datetime strings follow RFC 3339, section 5.6, and the longest possible
	// such string is 9999-12-31T23:59:59.999999999Z, which has length 30 - we add 2
	// to allow for quotation marks
	maxDatetimeSizeJSON = 32
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
			maxItems := int64(-1)
			if s.Items.ValueValidation != nil {
				if s.Items.ValueValidation.MaxItems != nil {
					maxItems = *s.Items.ValueValidation.MaxItems
				}
			}
			if maxItems == -1 {
				maxItems = estimateMaxSizeJSON(s)
			}
			if itemsType != nil {
				return NewListType(itemsType, maxItems)
			}
		}
		return nil
	case "object":
		if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
			propsType := SchemaDeclType(s.AdditionalProperties.Structural, s.AdditionalProperties.Structural.XEmbeddedResource)
			if propsType != nil {
				maxProperties := int64(-1)
				if s.ValueValidation != nil {
					if s.ValueValidation.MaxProperties != nil {
						maxProperties = *s.ValueValidation.MaxProperties
					}
				}
				if maxProperties == -1 {
					maxProperties = estimateMaxSizeJSON(s)
				}
				return NewMapType(StringType, propsType, maxProperties)
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
		if s.ValueValidation != nil && s.ValueValidation.Format != "" {
			switch s.ValueValidation.Format {
			case "byte":
				byteWithMaxLength := newSimpleType("bytes", decls.Bytes, types.Bytes([]byte{}))
				if s.ValueValidation.MaxLength != nil {
					byteWithMaxLength.MaxLength = *s.ValueValidation.MaxLength
				} else {
					byteWithMaxLength.MaxLength = estimateMaxSizeJSON(s)
				}
				return byteWithMaxLength
			case "duration":
				durationWithMaxLength := newSimpleType("duration", decls.Duration, types.Duration{Duration: time.Duration(0)})
				durationWithMaxLength.MaxLength = estimateMaxSizeJSON(s)
				return DurationType
			case "date", "date-time":
				timestampWithMaxLength := newSimpleType("timestamp", decls.Timestamp, types.Timestamp{Time: time.Time{}})
				timestampWithMaxLength.MaxLength = estimateMaxSizeJSON(s)
				return TimestampType
			}
		}
		strWithMaxLength := newSimpleType("string", decls.String, types.String(""))
		if s.ValueValidation != nil && s.ValueValidation.MaxLength != nil {
			strWithMaxLength.MaxLength = *s.ValueValidation.MaxLength
		} else {
			strWithMaxLength.MaxLength = estimateMaxSizeJSON(s)
		}
		return strWithMaxLength
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
		Generic:         s.Generic,
		Extensions:      s.Extensions,
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
			"name":         stringType,
			"generateName": stringType,
		},
	}
	result.Properties = props

	return result
}

// estimateMinSizeJSON estimates the minimum size in bytes of the given schema when serialized in JSON.
// minLength/minProperties/minItems is not currently taken into account, so the estimate is a little
// smaller than the actual smallest size.
func estimateMinSizeJSON(s *schema.Structural) int64 {
	switch s.Type {
	// take a comma into account for all cases since we'll generally call this function for arrays/maps
	case "boolean":
		// true,
		return 5
	case "number":
		// 0,
		return 2
	case "integer":
		// 0,
		return 2
	case "string":
		// "",
		return 3
	case "array":
		// [],
		return 3
	case "object":
		objSize := int64(3) // {},
		// sum of all non-optional properties
		if s.ValueValidation != nil {
			for _, propName := range s.ValueValidation.Required {
				prop := s.Properties[propName]
				// add 3, 2 for quotations around the property name and 1 for the colon
				objSize += int64(len(propName)) + estimateMinSizeJSON(&prop) + 3
			}
		}
		return objSize
	}
	// TODO(DangerOnTheRanger): better error handling (We should never get here in normal operation)
	return -1
}

// estimateMaxSizeJSON estimates the maximum number of elements that can fit in s considering request size
// constraints.
func estimateMaxSizeJSON(s *schema.Structural) int64 {
	switch s.Type {
	case "string":
		if s.ValueValidation != nil && s.ValueValidation.Format != "" && s.ValueValidation.Format != "byte" {
			switch s.Type {
			case "duration":
				return maxDurationSizeJSON
			case "date", "date-time":
				return maxDatetimeSizeJSON
			}
		}
		// subtract 2 to account for ""
		return (maxRequestSizeBytes - 2)
	case "array":
		// subtract 2 to account for [ and ]
		return (maxRequestSizeBytes - 2) / estimateMinSizeJSON(s.Items)
	case "object":
		if s.AdditionalProperties != nil && s.AdditionalProperties.Structural != nil {
			// smallest possible key ("") + colon + smallest possible value, realistically the actual keys
			// will all vary in length
			// TODO(DangerOnTheRanger): is there a way to calculate how many bytes unique keys will need?
			keyValuePairSize := estimateMinSizeJSON(s.AdditionalProperties.Structural) + 3
			// subtract 2 to account for { and }
			return (maxRequestSizeBytes - 2) / keyValuePairSize
		} else {
			// this codepath executes in the case of non-map objects,
			// but regular objects have no concept of maxProperties
			return -1
		}
	}
	// TODO(DangerOnTheRanger): better error handling (We should never get here in normal operation)
	return -1
}
