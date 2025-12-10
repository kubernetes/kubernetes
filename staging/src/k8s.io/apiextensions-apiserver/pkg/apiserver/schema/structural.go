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

import (
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// +k8s:deepcopy-gen=true

// Structural represents a structural schema.
type Structural struct {
	Items                *Structural
	Properties           map[string]Structural
	AdditionalProperties *StructuralOrBool

	Generic
	Extensions
	ValidationExtensions

	ValueValidation *ValueValidation
}

// +k8s:deepcopy-gen=true

// StructuralOrBool is either a structural schema or a boolean.
type StructuralOrBool struct {
	Structural *Structural
	Bool       bool
}

// +k8s:deepcopy-gen=true

// Generic contains the generic schema fields not allowed in value validation.
type Generic struct {
	Description string
	// type specifies the type of a value.
	// It can be object, array, number, integer, boolean, string.
	// It is optional only if x-kubernetes-preserve-unknown-fields
	// or x-kubernetes-int-or-string is true.
	Type     string
	Title    string
	Default  JSON
	Nullable bool
}

// +k8s:deepcopy-gen=true

// Extensions contains the Kubernetes OpenAPI v3 vendor extensions.
type Extensions struct {
	// x-kubernetes-preserve-unknown-fields stops the API server
	// decoding step from pruning fields which are not specified
	// in the validation schema. This affects fields recursively,
	// but switches back to normal pruning behaviour if nested
	// properties or additionalProperties are specified in the schema.
	// False means that the pruning behaviour is inherited from the parent.
	// False does not mean to activate pruning.
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
	//    - ... zero or more
	XIntOrString bool

	// x-kubernetes-list-map-keys annotates lists with the x-kubernetes-list-type `map` by specifying the keys used
	// as the index of the map.
	//
	// This tag MUST only be used on lists that have the "x-kubernetes-list-type"
	// extension set to "map". Also, the values specified for this attribute must
	// be a scalar typed field of the child structure (no nesting is supported).
	XListMapKeys []string

	// x-kubernetes-list-type annotates a list to further describe its topology.
	// This extension must only be used on lists and may have 3 possible values:
	//
	// 1) `atomic`: the list is treated as a single entity, like a scalar.
	//      Atomic lists will be entirely replaced when updated. This extension
	//      may be used on any type of list (struct, scalar, ...).
	// 2) `set`:
	//      Sets are lists that must not have multiple items with the same value. Each
	//      value must be a scalar (or another atomic type).
	// 3) `map`:
	//      These lists are like maps in that their elements have a non-index key
	//      used to identify them. Order is preserved upon merge. The map tag
	//      must only be used on a list with elements of type object.
	XListType *string

	// x-kubernetes-map-type annotates an object to further describe its topology.
	// This extension must only be used when type is object and may have 2 possible values:
	//
	// 1) `granular`:
	//      These maps are actual maps (key-value pairs) and each fields are independent
	//      from each other (they can each be manipulated by separate actors). This is
	//      the default behaviour for all maps.
	// 2) `atomic`: the list is treated as a single entity, like a scalar.
	//      Atomic maps will be entirely replaced when updated.
	// +optional
	XMapType *string
}

// +k8s:deepcopy-gen=true

// ValidationExtensions contains the Kubernetes OpenAPI v3 extensions that are
// used for validation rather than structure.
type ValidationExtensions struct {
	// x-kubernetes-validations describes a list of validation rules for expression validation.
	// Use the v1 struct since this gets serialized as an extension.
	XValidations apiextensionsv1.ValidationRules
}

// +k8s:deepcopy-gen=true

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
	Enum             []JSON
	MaxProperties    *int64
	MinProperties    *int64
	Required         []string
	AllOf            []NestedValueValidation
	OneOf            []NestedValueValidation
	AnyOf            []NestedValueValidation
	Not              *NestedValueValidation
}

// +k8s:deepcopy-gen=true

// NestedValueValidation contains value validations, items and properties usable when nested
// under a logical junctor, and catch all structs for generic and vendor extensions schema fields.
type NestedValueValidation struct {
	ValueValidation
	ValidationExtensions

	Items                *NestedValueValidation
	Properties           map[string]NestedValueValidation
	AdditionalProperties *NestedValueValidation

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
	//    - ... zero or more
	ForbiddenGenerics   Generic
	ForbiddenExtensions Extensions
}

// JSON wraps an arbitrary JSON value to be able to implement deepcopy.
type JSON struct {
	Object interface{}
}

// DeepCopy creates a deep copy of the wrapped JSON value.
func (j JSON) DeepCopy() JSON {
	return JSON{runtime.DeepCopyJSONValue(j.Object)}
}

// DeepCopyInto creates a deep copy of the wrapped JSON value and stores it in into.
func (j JSON) DeepCopyInto(into *JSON) {
	into.Object = runtime.DeepCopyJSONValue(j.Object)
}
