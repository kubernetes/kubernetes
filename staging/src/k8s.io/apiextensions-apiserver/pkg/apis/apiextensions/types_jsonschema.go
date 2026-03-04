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

package apiextensions

// FieldValueErrorReason is a machine-readable value providing more detail about why a field failed the validation.
// +enum
type FieldValueErrorReason string

const (
	// FieldValueRequired is used to report required values that are not
	// provided (e.g. empty strings, null values, or empty arrays).
	FieldValueRequired FieldValueErrorReason = "FieldValueRequired"
	// FieldValueDuplicate is used to report collisions of values that must be
	// unique (e.g. unique IDs).
	FieldValueDuplicate FieldValueErrorReason = "FieldValueDuplicate"
	// FieldValueInvalid is used to report malformed values (e.g. failed regex
	// match, too long, out of bounds).
	FieldValueInvalid FieldValueErrorReason = "FieldValueInvalid"
	// FieldValueForbidden is used to report valid (as per formatting rules)
	// values which would be accepted under some conditions, but which are not
	// permitted by the current conditions (such as security policy).
	FieldValueForbidden FieldValueErrorReason = "FieldValueForbidden"
)

// JSONSchemaProps is a JSON-Schema following Specification Draft 4 (http://json-schema.org/).
type JSONSchemaProps struct {
	ID                   string
	Schema               JSONSchemaURL
	Ref                  *string
	Description          string
	Type                 string
	Nullable             bool
	Format               string
	Title                string
	Default              *JSON
	Maximum              *float64
	ExclusiveMaximum     bool
	Minimum              *float64
	ExclusiveMinimum     bool
	MaxLength            *int64
	MinLength            *int64
	Pattern              string
	MaxItems             *int64
	MinItems             *int64
	UniqueItems          bool
	MultipleOf           *float64
	Enum                 []JSON
	MaxProperties        *int64
	MinProperties        *int64
	Required             []string
	Items                *JSONSchemaPropsOrArray
	AllOf                []JSONSchemaProps
	OneOf                []JSONSchemaProps
	AnyOf                []JSONSchemaProps
	Not                  *JSONSchemaProps
	Properties           map[string]JSONSchemaProps
	AdditionalProperties *JSONSchemaPropsOrBool
	PatternProperties    map[string]JSONSchemaProps
	Dependencies         JSONSchemaDependencies
	AdditionalItems      *JSONSchemaPropsOrBool
	Definitions          JSONSchemaDefinitions
	ExternalDocs         *ExternalDocumentation
	Example              *JSON

	// x-kubernetes-preserve-unknown-fields stops the API server
	// decoding step from pruning fields which are not specified
	// in the validation schema. This affects fields recursively,
	// but switches back to normal pruning behaviour if nested
	// properties or additionalProperties are specified in the schema.
	// This can either be true or undefined. False is forbidden.
	XPreserveUnknownFields *bool

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

	// x-kubernetes-list-map-keys annotates an array with the x-kubernetes-list-type `map` by specifying the keys used
	// as the index of the map.
	//
	// This tag MUST only be used on lists that have the "x-kubernetes-list-type"
	// extension set to "map". Also, the values specified for this attribute must
	// be a scalar typed field of the child structure (no nesting is supported).
	XListMapKeys []string

	// x-kubernetes-list-type annotates an array to further describe its topology.
	// This extension must only be used on lists and may have 3 possible values:
	//
	// 1) `atomic`: the list is treated as a single entity, like a scalar.
	//      Atomic lists will be entirely replaced when updated. This extension
	//      may be used on any type of list (struct, scalar, ...).
	// 2) `set`:
	//      Sets are lists that must not have multiple items with the same value. Each
	//      value must be a scalar, an object with x-kubernetes-map-type `atomic` or an
	//      array with x-kubernetes-list-type `atomic`.
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

	// x-kubernetes-validations -kubernetes-validations describes a list of validation rules written in the CEL expression language.
	// +patchMergeKey=rule
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=rule
	XValidations ValidationRules
}

// ValidationRules describes a list of validation rules written in the CEL expression language.
type ValidationRules []ValidationRule

// ValidationRule describes a validation rule written in the CEL expression language.
type ValidationRule struct {
	// Rule represents the expression which will be evaluated by CEL.
	// ref: https://github.com/google/cel-spec
	// The Rule is scoped to the location of the x-kubernetes-validations extension in the schema.
	// The `self` variable in the CEL expression is bound to the scoped value.
	// Example:
	// - Rule scoped to the root of a resource with a status subresource: {"rule": "self.status.actual <= self.spec.maxDesired"}
	//
	// If the Rule is scoped to an object with properties, the accessible properties of the object are field selectable
	// via `self.field` and field presence can be checked via `has(self.field)`. Null valued fields are treated as
	// absent fields in CEL expressions.
	// If the Rule is scoped to an object with additionalProperties (i.e. a map) the value of the map
	// are accessible via `self[mapKey]`, map containment can be checked via `mapKey in self` and all entries of the map
	// are accessible via CEL macros and functions such as `self.all(...)`.
	// If the Rule is scoped to an array, the elements of the array are accessible via `self[i]` and also by macros and
	// functions.
	// If the Rule is scoped to a scalar, `self` is bound to the scalar value.
	// Examples:
	// - Rule scoped to a map of objects: {"rule": "self.components['Widget'].priority < 10"}
	// - Rule scoped to a list of integers: {"rule": "self.values.all(value, value >= 0 && value < 100)"}
	// - Rule scoped to a string value: {"rule": "self.startsWith('kube')"}
	//
	// The `apiVersion`, `kind`, `metadata.name` and `metadata.generateName` are always accessible from the root of the
	// object and from any x-kubernetes-embedded-resource annotated objects. No other metadata properties are accessible.
	//
	// Unknown data preserved in custom resources via x-kubernetes-preserve-unknown-fields is not accessible in CEL
	// expressions. This includes:
	// - Unknown field values that are preserved by object schemas with x-kubernetes-preserve-unknown-fields.
	// - Object properties where the property schema is of an "unknown type". An "unknown type" is recursively defined as:
	//   - A schema with no type and x-kubernetes-preserve-unknown-fields set to true
	//   - An array where the items schema is of an "unknown type"
	//   - An object where the additionalProperties schema is of an "unknown type"
	//
	// Only property names of the form `[a-zA-Z_.-/][a-zA-Z0-9_.-/]*` are accessible.
	// Accessible property names are escaped according to the following rules when accessed in the expression:
	// - '__' escapes to '__underscores__'
	// - '.' escapes to '__dot__'
	// - '-' escapes to '__dash__'
	// - '/' escapes to '__slash__'
	// - Property names that exactly match a CEL RESERVED keyword escape to '__{keyword}__'. The keywords are:
	//	  "true", "false", "null", "in", "as", "break", "const", "continue", "else", "for", "function", "if",
	//	  "import", "let", "loop", "package", "namespace", "return".
	// Examples:
	//   - Rule accessing a property named "namespace": {"rule": "self.__namespace__ > 0"}
	//   - Rule accessing a property named "x-prop": {"rule": "self.x__dash__prop > 0"}
	//   - Rule accessing a property named "redact__d": {"rule": "self.redact__underscores__d > 0"}
	//
	// Equality on arrays with x-kubernetes-list-type of 'set' or 'map' ignores element order, i.e. [1, 2] == [2, 1].
	// Concatenation on arrays with x-kubernetes-list-type use the semantics of the list type:
	//   - 'set': `X + Y` performs a union where the array positions of all elements in `X` are preserved and
	//     non-intersecting elements in `Y` are appended, retaining their partial order.
	//   - 'map': `X + Y` performs a merge where the array positions of all keys in `X` are preserved but the values
	//     are overwritten by values in `Y` when the key sets of `X` and `Y` intersect. Elements in `Y` with
	//     non-intersecting keys are appended, retaining their partial order.
	//
	// If `rule` makes use of the `oldSelf` variable it is implicitly a
	// `transition rule`.
	//
	// By default, the `oldSelf` variable is the same type as `self`.
	// When `optionalOldSelf` is true, the `oldSelf` variable is a CEL optional
	//  variable whose value() is the same type as `self`.
	// See the documentation for the `optionalOldSelf` field for details.
	//
	// Transition rules by default are applied only on UPDATE requests and are
	// skipped if an old value could not be found. You can opt a transition
	// rule into unconditional evaluation by setting `optionalOldSelf` to true.
	//
	Rule string
	// Message represents the message displayed when validation fails. The message is required if the Rule contains
	// line breaks. The message must not contain line breaks.
	// If unset, the message is "failed rule: {Rule}".
	// e.g. "must be a URL with the host matching spec.host"
	Message string
	// MessageExpression declares a CEL expression that evaluates to the validation failure message that is returned when this rule fails.
	// Since messageExpression is used as a failure message, it must evaluate to a string.
	// If both message and messageExpression are present on a rule, then messageExpression will be used if validation
	// fails. If messageExpression results in a runtime error, the runtime error is logged, and the validation failure message is produced
	// as if the messageExpression field were unset. If messageExpression evaluates to an empty string, a string with only spaces, or a string
	// that contains line breaks, then the validation failure message will also be produced as if the messageExpression field were unset, and
	// the fact that messageExpression produced an empty string/string with only spaces/string with line breaks will be logged.
	// messageExpression has access to all the same variables as the rule; the only difference is the return type.
	// Example:
	// "x must be less than max ("+string(self.max)+")"
	// +optional
	MessageExpression string
	// reason provides a machine-readable validation failure reason that is returned to the caller when a request fails this validation rule.
	// The HTTP status code returned to the caller will match the reason of the reason of the first failed validation rule.
	// The currently supported reasons are: "FieldValueInvalid", "FieldValueForbidden", "FieldValueRequired", "FieldValueDuplicate".
	// If not set, default to use "FieldValueInvalid".
	// All future added reasons must be accepted by clients when reading this value and unknown reasons should be treated as FieldValueInvalid.
	// +optional
	Reason *FieldValueErrorReason
	// fieldPath represents the field path returned when the validation fails.
	// It must be a relative JSON path (i.e. with array notation) scoped to the location of this x-kubernetes-validations extension in the schema and refer to an existing field.
	// e.g. when validation checks if a specific attribute `foo` under a map `testMap`, the fieldPath could be set to `.testMap.foo`
	// If the validation checks two lists must have unique attributes, the fieldPath could be set to either of the list: e.g. `.testList`
	// It does not support list numeric index.
	// It supports child operation to refer to an existing field currently. Refer to [JSONPath support in Kubernetes](https://kubernetes.io/docs/reference/kubectl/jsonpath/) for more info.
	// Numeric index of array is not supported.
	// For field name which contains special characters, use `['specialName']` to refer the field name.
	// e.g. for attribute `foo.34$` appears in a list `testList`, the fieldPath could be set to `.testList['foo.34$']`
	// +optional
	FieldPath string

	// optionalOldSelf is used to opt a transition rule into evaluation
	// even when the object is first created, or if the old object is
	// missing the value.
	//
	// When enabled `oldSelf` will be a CEL optional whose value will be
	// `None` if there is no old value, or when the object is initially created.
	//
	// You may check for presence of oldSelf using `oldSelf.hasValue()` and
	// unwrap it after checking using `oldSelf.value()`. Check the CEL
	// documentation for Optional types for more information:
	// https://pkg.go.dev/github.com/google/cel-go/cel#OptionalTypes
	//
	// May not be set unless `oldSelf` is used in `rule`.
	//
	// +featureGate=CRDValidationRatcheting
	// +optional
	OptionalOldSelf *bool
}

// JSON represents any valid JSON value.
// These types are supported: bool, int64, float64, string, []interface{}, map[string]interface{} and nil.
type JSON interface{}

// JSONSchemaURL represents a schema url.
type JSONSchemaURL string

// JSONSchemaPropsOrArray represents a value that can either be a JSONSchemaProps
// or an array of JSONSchemaProps. Mainly here for serialization purposes.
type JSONSchemaPropsOrArray struct {
	Schema      *JSONSchemaProps
	JSONSchemas []JSONSchemaProps
}

// JSONSchemaPropsOrBool represents JSONSchemaProps or a boolean value.
// Defaults to true for the boolean property.
type JSONSchemaPropsOrBool struct {
	Allows bool
	Schema *JSONSchemaProps
}

// JSONSchemaDependencies represent a dependencies property.
type JSONSchemaDependencies map[string]JSONSchemaPropsOrStringArray

// JSONSchemaPropsOrStringArray represents a JSONSchemaProps or a string array.
type JSONSchemaPropsOrStringArray struct {
	Schema   *JSONSchemaProps
	Property []string
}

// JSONSchemaDefinitions contains the models explicitly defined in this spec.
type JSONSchemaDefinitions map[string]JSONSchemaProps

// ExternalDocumentation allows referencing an external resource for extended documentation.
type ExternalDocumentation struct {
	Description string
	URL         string
}
