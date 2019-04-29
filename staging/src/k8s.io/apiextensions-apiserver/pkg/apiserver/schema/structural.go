/*
Copyright 2019 The Kubernetes Authors.

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

package schema

// Structural represents a structural schema.
type Structural struct {
	Items      *Structural
	Properties map[string]Structural

	Generic
	Extensions

	*ValueValidation
}

// StructuralOrBool is either a structural schema or a boolean.
type StructuralOrBool struct {
	Structural *Structural
	Bool       bool
}

// Generic contains the generic schema fields not allowed in value validation.
type Generic struct {
	// type specifies the type of a value.
	// It can be object, array, number, integer, boolean, string.
	// It is optional only if x-kubernetes-preserve-unknown-fields
	// or x-kubernetes-int-or-string is true.
	Type                 string
	AdditionalProperties *StructuralOrBool

	Description string
	Title       string
	Nullable    bool
	Default     interface{}
}

// Extensions contains the Kubernetes OpenAPI v3 vendor extensions.
type Extensions struct {
	// x-kubernetes-preserve-unknown-fields stops the API server
	// decoding step from pruning fields which are not specified
	// in the validation schema. This affects fields recursively,
	// but switches back to normal pruning behaviour if nested
	// properties or additionalProperties are specified in the schema.
	XPreserveUnknownFields bool

	// x-kubernetes-embedded-resource defines that the value is an
	// embedded Kubernetes runtime.Object, with TypeMeta and
	// ObjectMeta. The type must be object. It is allowed to further
	// restrict the embedded object. Both ObjectMeta and TypeMeta
	// are validated automatically. x-kubernetes-preserve-unknown-fields
	// must be true.
	XEmbeddedResource bool

	// x-kubernetes-int-or-string specifies that this value is
	// either an integer or a string. If this is true, an empty
	// type is allowed and type as child of anyOf is permitted
	// if following one of the following patterns:
	//
	// 1) anyOf:
	//    - type: integer
	//    - type: string
	// 2) allOf:
	//    - anyOf:
	//      - type: integer
	//      - type: string
	//    - <NestedValueValidation>
	XIntOrString bool
}

// ValueValidation contains all schema fields not contributing to the structure of the schema.
type ValueValidation struct {
	Format           string
	Maximum          *float64
	ExclusiveMaximum bool
	Minimum          *float64
	ExclusiveMinimum bool
	MaxLength        *int64
	MinLength        *int64
	Pattern          string
	MaxItems         *int64
	MinItems         *int64
	UniqueItems      bool
	MultipleOf       *float64
	Enum             []interface{}
	MaxProperties    *int64
	MinProperties    *int64
	Required         []string
	AllOf            []NestedValueValidation
	OneOf            []NestedValueValidation
	AnyOf            []NestedValueValidation
	Not              *NestedValueValidation
}

// NestedValueValidation contains value validations, items and properties usable when nested
// under a logical junctor, and catch all structs for generic and vendor extensions schema fields.
type NestedValueValidation struct {
	ValueValidation

	Items      *NestedValueValidation
	Properties map[string]NestedValueValidation

	// Anything set in the following will make the scheme
	// non-structural, with the exception of these two patterns if
	// x-kubernetes-int-or-string is true:
	//
	// 1) anyOf:
	//    - type: integer
	//    - type: string
	// 2) allOf:
	//    - anyOf:
	//      - type: integer
	//      - type: string
	//    - <NestedValueValidation>
	ForbiddenGenerics   Generic
	ForbiddenExtensions Extensions
}
