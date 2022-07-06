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
	// TODO(DangerOnTheRanger): wire in MaxRequestBodyBytes from apiserver/pkg/server/options/server_run_options.go to make this configurable
	maxRequestSizeBytes = int64(3 * 1024 * 1024)
	// OpenAPI duration strings follow RFC 3339, section 5.6 - see the comment on maxDatetimeSizeJSON
	maxDurationSizeJSON = 32
	// OpenAPI datetime strings follow RFC 3339, section 5.6, and the longest possible
	// such string is 9999-12-31T23:59:59.999999999Z, which has length 30 - we add 2
	// to allow for quotation marks
	maxDatetimeSizeJSON = 32
	// Golang allows a string of 0 to be parsed as a duration, so that plus 2 to account for
	// quotation marks makes 3
	minDurationSizeJSON = 3
	// RFC 3339 dates require YYYY-MM-DD, and then we add 2 to allow for quotation marks
	dateSizeJSON = 12
	// RFC 3339 times require 2-digit 24-hour time at the very least plus a capital T at the start,
	// e.g., T23, and we add 2 to allow for quotation marks as usual
	minTimeSizeJSON = 5
	// RFC 3339 datetimes require a full date (YYYY-MM-DD) and full time (HH:MM:SS), and we add 3 for
	// quotation marks like always in addition to the capital T that separates the date and time
	minDatetimeSizeJSON = 21
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
		dyn := newSimpleType("dyn", decls.Dyn, nil)
		// handle x-kubernetes-int-or-string by returning the max length of the largest possible string
		dyn.MaxElements = maxRequestSizeBytes - 2
		return dyn
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
			var maxItems int64
			if s.ValueValidation != nil && s.ValueValidation.MaxItems != nil {
				maxItems = zeroIfNegative(*s.ValueValidation.MaxItems)
			} else {
				maxItems = estimateMaxArrayItemsPerRequest(s.Items)
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
				var maxProperties int64
				if s.ValueValidation != nil && s.ValueValidation.MaxProperties != nil {
					maxProperties = zeroIfNegative(*s.ValueValidation.MaxProperties)
				} else {
					maxProperties = estimateMaxAdditionalPropertiesPerRequest(s.AdditionalProperties.Structural)
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
		if s.ValueValidation != nil {
			switch s.ValueValidation.Format {
			case "byte":
				byteWithMaxLength := newSimpleType("bytes", decls.Bytes, types.Bytes([]byte{}))
				if s.ValueValidation.MaxLength != nil {
					byteWithMaxLength.MaxElements = zeroIfNegative(*s.ValueValidation.MaxLength)
				} else {
					byteWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
				}
				return byteWithMaxLength
			case "duration":
				durationWithMaxLength := newSimpleType("duration", decls.Duration, types.Duration{Duration: time.Duration(0)})
				durationWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
				return durationWithMaxLength
			case "date", "date-time":
				timestampWithMaxLength := newSimpleType("timestamp", decls.Timestamp, types.Timestamp{Time: time.Time{}})
				timestampWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
				return timestampWithMaxLength
			}
		}
		strWithMaxLength := newSimpleType("string", decls.String, types.String(""))
		if s.ValueValidation != nil && s.ValueValidation.MaxLength != nil {
			// multiply the user-provided max length by 4 in the case of an otherwise-untyped string
			// we do this because the OpenAPIv3 spec indicates that maxLength is specified in runes/code points,
			// but we need to reason about length for things like request size, so we use bytes in this code (and an individual
			// unicode code point can be up to 4 bytes long)
			strWithMaxLength.MaxElements = zeroIfNegative(*s.ValueValidation.MaxLength) * 4
		} else {
			strWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
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

func zeroIfNegative(v int64) int64 {
	if v < 0 {
		return 0
	}
	return v
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

// MaxCardinality returns the maximum number of times data conforming to the schema could possibly exist in
// an object serialized to JSON. For cases where a schema is contained under map or array schemas of unbounded
// size, this can be used as an estimate as the worst case number of times data matching the schema could be repeated.
// Note that this only assumes a single comma between data elements, so if the schema is contained under only maps,
// this estimates a higher cardinality that would be possible.
func MaxCardinality(s *schema.Structural) uint64 {
	sz := estimateMinSizeJSON(s) + 1 // assume at least one comma between elements
	return uint64(maxRequestSizeBytes / sz)
}

// estimateMinSizeJSON estimates the minimum size in bytes of the given schema when serialized in JSON.
// minLength/minProperties/minItems are not currently taken into account, so if these limits are set the
// minimum size might be higher than what estimateMinSizeJSON returns.
func estimateMinSizeJSON(s *schema.Structural) int64 {
	if s == nil {
		// minimum valid JSON token has length 1 (single-digit number like `0`)
		return 1
	}
	switch s.Type {
	case "boolean":
		// true
		return 4
	case "number", "integer":
		// 0
		return 1
	case "string":
		if s.ValueValidation != nil {
			switch s.ValueValidation.Format {
			case "duration":
				return minDurationSizeJSON
			case "date":
				return dateSizeJSON
			case "date-time":
				return minDatetimeSizeJSON
			}
		}
		// ""
		return 2
	case "array":
		// []
		return 2
	case "object":
		// {}
		objSize := int64(2)
		// exclude optional fields since the request can omit them
		if s.ValueValidation != nil {
			for _, propName := range s.ValueValidation.Required {
				if prop, ok := s.Properties[propName]; ok {
					if prop.Default.Object != nil {
						// exclude fields with a default, those are filled in server-side
						continue
					}
					// add 4, 2 for quotations around the property name, 1 for the colon, and 1 for a comma
					objSize += int64(len(propName)) + estimateMinSizeJSON(&prop) + 4
				}
			}
		}
		return objSize
	}
	if s.XIntOrString {
		// 0
		return 1
	}
	// this code should be unreachable, so return the safest possible value considering this can be used as
	// a divisor
	return 1
}

// estimateMaxArrayItemsPerRequest estimates the maximum number of array items with
// the provided schema that can fit into a single request.
func estimateMaxArrayItemsPerRequest(itemSchema *schema.Structural) int64 {
	// subtract 2 to account for [ and ]
	return (maxRequestSizeBytes - 2) / (estimateMinSizeJSON(itemSchema) + 1)
}

// estimateMaxStringLengthPerRequest estimates the maximum string length (in characters)
// of a string compatible with the format requirements in the provided schema.
// must only be called on schemas of type "string" or x-kubernetes-int-or-string: true
func estimateMaxStringLengthPerRequest(s *schema.Structural) int64 {
	if s.ValueValidation == nil || s.XIntOrString {
		// subtract 2 to account for ""
		return (maxRequestSizeBytes - 2)
	}
	switch s.ValueValidation.Format {
	case "duration":
		return maxDurationSizeJSON
	case "date":
		return dateSizeJSON
	case "date-time":
		return maxDatetimeSizeJSON
	default:
		// subtract 2 to account for ""
		return (maxRequestSizeBytes - 2)
	}
}

// estimateMaxAdditionalPropertiesPerRequest estimates the maximum number of additional properties
// with the provided schema that can fit into a single request.
func estimateMaxAdditionalPropertiesPerRequest(additionalPropertiesSchema *schema.Structural) int64 {
	// 2 bytes for key + "" + colon + comma + smallest possible value, realistically the actual keys
	// will all vary in length
	keyValuePairSize := estimateMinSizeJSON(additionalPropertiesSchema) + 6
	// subtract 2 to account for { and }
	return (maxRequestSizeBytes - 2) / keyValuePairSize
}
