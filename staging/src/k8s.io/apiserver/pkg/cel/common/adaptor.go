/*
Copyright 2023 The Kubernetes Authors.

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

// Schema is the adapted type for an OpenAPI schema that CEL uses.
// This schema does not cover all OpenAPI fields but only these CEL requires
// are exposed as getters.
type Schema interface {
	// Type returns the OpenAPI type.
	// Multiple types are not supported. It should return
	// empty string if no type is specified.
	Type() string

	// Format returns the OpenAPI format. May be empty
	Format() string

	// Items returns the OpenAPI items. or nil of this field does not exist or
	// contains no schema.
	Items() Schema

	// Properties returns the OpenAPI properties, or nil if this field does not
	// exist.
	// The values of the returned map are of the adapted type.
	Properties() map[string]Schema

	// AdditionalProperties returns the OpenAPI additional properties field,
	// or nil if this field does not exist.
	AdditionalProperties() SchemaOrBool

	// Default returns the OpenAPI default field, or nil if this field does not exist.
	Default() any

	Validations
	KubeExtensions

	// WithTypeAndObjectMeta returns a schema that has the type and object meta set.
	// the type includes "kind", "apiVersion" field
	// the "metadata" field requires "name" and "generateName" to be set
	// The original schema must not be mutated. Make a copy if necessary.
	WithTypeAndObjectMeta() Schema
}

// Validations contains OpenAPI validation that the CEL library uses.
type Validations interface {
	Pattern() string
	Minimum() *float64
	IsExclusiveMinimum() bool
	Maximum() *float64
	IsExclusiveMaximum() bool
	MultipleOf() *float64
	MinItems() *int64
	MaxItems() *int64
	MinLength() *int64
	MaxLength() *int64
	MinProperties() *int64
	MaxProperties() *int64
	Required() []string
	Enum() []any
	Nullable() bool
	UniqueItems() bool

	AllOf() []Schema
	OneOf() []Schema
	AnyOf() []Schema
	Not() Schema
}

// KubeExtensions contains Kubernetes-specific extensions to the OpenAPI schema.
type KubeExtensions interface {
	IsXIntOrString() bool
	IsXEmbeddedResource() bool
	IsXPreserveUnknownFields() bool
	XListType() string
	XListMapKeys() []string
	XMapType() string
	XValidations() []ValidationRule
}

// ValidationRule represents a single x-kubernetes-validations rule.
type ValidationRule interface {
	Rule() string
	Message() string
	MessageExpression() string
	FieldPath() string
}

// SchemaOrBool contains either a schema or a boolean indicating if the object
// can contain any fields.
type SchemaOrBool interface {
	Schema() Schema
	Allows() bool
}
