/*
Copyright 2022 The Kubernetes Authors.

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

package common

import (
	"time"

	"github.com/google/cel-go/cel"
	"github.com/google/cel-go/common/types"

	apiservercel "k8s.io/apiserver/pkg/cel"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const maxRequestSizeBytes = apiservercel.DefaultMaxRequestSizeBytes

// SchemaDeclType converts the structural schema to a CEL declaration, or returns nil if the
// structural schema should not be exposed in CEL expressions.
// Set isResourceRoot to true for the root of a custom resource or embedded resource.
//
// Schemas with XPreserveUnknownFields not exposed unless they are objects. Array and "maps" schemas
// are not exposed if their items or additionalProperties schemas are not exposed. Object Properties are not exposed
// if their schema is not exposed.
//
// The CEL declaration for objects with XPreserveUnknownFields does not expose unknown fields.
func SchemaDeclType(s Schema, isResourceRoot bool) *apiservercel.DeclType {
	if s == nil {
		return nil
	}
	if s.IsXIntOrString() {
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
		dyn := apiservercel.NewSimpleTypeWithMinSize("dyn", cel.DynType, nil, 1) // smallest value for a serialized x-kubernetes-int-or-string is 0
		// handle x-kubernetes-int-or-string by returning the max length/min serialized size of the largest possible string
		dyn.MaxElements = maxRequestSizeBytes - 2
		return dyn
	}

	// We ignore XPreserveUnknownFields since we don't support validation rules on
	// data that we don't have schema information for.

	if isResourceRoot {
		// 'apiVersion', 'kind', 'metadata.name' and 'metadata.generateName' are always accessible to validator rules
		// at the root of resources, even if not specified in the schema.
		// This includes the root of a custom resource and the root of XEmbeddedResource objects.
		s = s.WithTypeAndObjectMeta()
	}

	switch s.Type() {
	case "array":
		if s.Items() != nil {
			itemsType := SchemaDeclType(s.Items(), s.Items().IsXEmbeddedResource())
			if itemsType == nil {
				return nil
			}
			var maxItems int64
			if s.MaxItems() != nil {
				maxItems = zeroIfNegative(*s.MaxItems())
			} else {
				maxItems = estimateMaxArrayItemsFromMinSize(itemsType.MinSerializedSize)
			}
			return apiservercel.NewListType(itemsType, maxItems)
		}
		return nil
	case "object":
		if s.AdditionalProperties() != nil && s.AdditionalProperties().Schema() != nil {
			propsType := SchemaDeclType(s.AdditionalProperties().Schema(), s.AdditionalProperties().Schema().IsXEmbeddedResource())
			if propsType != nil {
				var maxProperties int64
				if s.MaxProperties() != nil {
					maxProperties = zeroIfNegative(*s.MaxProperties())
				} else {
					maxProperties = estimateMaxAdditionalPropertiesFromMinSize(propsType.MinSerializedSize)
				}
				return apiservercel.NewMapType(apiservercel.StringType, propsType, maxProperties)
			}
			return nil
		}
		fields := make(map[string]*apiservercel.DeclField, len(s.Properties()))

		required := map[string]bool{}
		if s.Required() != nil {
			for _, f := range s.Required() {
				required[f] = true
			}
		}
		// an object will always be serialized at least as {}, so account for that
		minSerializedSize := int64(2)
		for name, prop := range s.Properties() {
			var enumValues []interface{}
			if prop.Enum() != nil {
				for _, e := range prop.Enum() {
					enumValues = append(enumValues, e)
				}
			}
			if fieldType := SchemaDeclType(prop, prop.IsXEmbeddedResource()); fieldType != nil {
				if propName, ok := apiservercel.Escape(name); ok {
					fields[propName] = apiservercel.NewDeclField(propName, fieldType, required[name], enumValues, prop.Default())
				}
				// the min serialized size for an object is 2 (for {}) plus the min size of all its required
				// properties
				// only include required properties without a default value; default values are filled in
				// server-side
				if required[name] && prop.Default() == nil {
					minSerializedSize += int64(len(name)) + fieldType.MinSerializedSize + 4
				}
			}
		}
		objType := apiservercel.NewObjectType("object", fields)
		objType.MinSerializedSize = minSerializedSize
		return objType
	case "string":
		switch s.Format() {
		case "byte":
			byteWithMaxLength := apiservercel.NewSimpleTypeWithMinSize("bytes", cel.BytesType, types.Bytes([]byte{}), apiservercel.MinStringSize)
			if s.MaxLength() != nil {
				byteWithMaxLength.MaxElements = zeroIfNegative(*s.MaxLength())
			} else {
				byteWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
			}
			return byteWithMaxLength
		case "duration":
			durationWithMaxLength := apiservercel.NewSimpleTypeWithMinSize("duration", cel.DurationType, types.Duration{Duration: time.Duration(0)}, int64(apiservercel.MinDurationSizeJSON))
			durationWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
			return durationWithMaxLength
		case "date":
			timestampWithMaxLength := apiservercel.NewSimpleTypeWithMinSize("timestamp", cel.TimestampType, types.Timestamp{Time: time.Time{}}, int64(apiservercel.JSONDateSize))
			timestampWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
			return timestampWithMaxLength
		case "date-time":
			timestampWithMaxLength := apiservercel.NewSimpleTypeWithMinSize("timestamp", cel.TimestampType, types.Timestamp{Time: time.Time{}}, int64(apiservercel.MinDatetimeSizeJSON))
			timestampWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
			return timestampWithMaxLength
		}

		strWithMaxLength := apiservercel.NewSimpleTypeWithMinSize("string", cel.StringType, types.String(""), apiservercel.MinStringSize)
		if s.MaxLength() != nil {
			// multiply the user-provided max length by 4 in the case of an otherwise-untyped string
			// we do this because the OpenAPIv3 spec indicates that maxLength is specified in runes/code points,
			// but we need to reason about length for things like request size, so we use bytes in this code (and an individual
			// unicode code point can be up to 4 bytes long)
			strWithMaxLength.MaxElements = zeroIfNegative(*s.MaxLength()) * 4
		} else {
			strWithMaxLength.MaxElements = estimateMaxStringLengthPerRequest(s)
		}
		return strWithMaxLength
	case "boolean":
		return apiservercel.BoolType
	case "number":
		return apiservercel.DoubleType
	case "integer":
		return apiservercel.IntType
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
func WithTypeAndObjectMeta(s *spec.Schema) *spec.Schema {
	if s.Properties != nil &&
		s.Properties["kind"].Type.Contains("string") &&
		s.Properties["apiVersion"].Type.Contains("string") &&
		s.Properties["metadata"].Type.Contains("object") &&
		s.Properties["metadata"].Properties != nil &&
		s.Properties["metadata"].Properties["name"].Type.Contains("string") &&
		s.Properties["metadata"].Properties["generateName"].Type.Contains("string") {
		return s
	}
	result := *s
	props := make(map[string]spec.Schema, len(s.Properties))
	for k, prop := range s.Properties {
		props[k] = prop
	}
	stringType := spec.StringProperty()
	props["kind"] = *stringType
	props["apiVersion"] = *stringType
	props["metadata"] = spec.Schema{
		SchemaProps: spec.SchemaProps{
			Type: []string{"object"},
			Properties: map[string]spec.Schema{
				"name":         *stringType,
				"generateName": *stringType,
			},
		},
	}
	result.Properties = props

	return &result
}

// estimateMaxStringLengthPerRequest estimates the maximum string length (in characters)
// of a string compatible with the format requirements in the provided schema.
// must only be called on schemas of type "string" or x-kubernetes-int-or-string: true
func estimateMaxStringLengthPerRequest(s Schema) int64 {
	if s.IsXIntOrString() {
		return maxRequestSizeBytes - 2
	}
	switch s.Format() {
	case "duration":
		return apiservercel.MaxDurationSizeJSON
	case "date":
		return apiservercel.JSONDateSize
	case "date-time":
		return apiservercel.MaxDatetimeSizeJSON
	default:
		// subtract 2 to account for ""
		return maxRequestSizeBytes - 2
	}
}

// estimateMaxArrayItemsPerRequest estimates the maximum number of array items with
// the provided minimum serialized size that can fit into a single request.
func estimateMaxArrayItemsFromMinSize(minSize int64) int64 {
	// subtract 2 to account for [ and ]
	return (maxRequestSizeBytes - 2) / (minSize + 1)
}

// estimateMaxAdditionalPropertiesPerRequest estimates the maximum number of additional properties
// with the provided minimum serialized size that can fit into a single request.
func estimateMaxAdditionalPropertiesFromMinSize(minSize int64) int64 {
	// 2 bytes for key + "" + colon + comma + smallest possible value, realistically the actual keys
	// will all vary in length
	keyValuePairSize := minSize + 6
	// subtract 2 to account for { and }
	return (maxRequestSizeBytes - 2) / keyValuePairSize
}
