// Copyright 2015 xeipuuv ( https://github.com/xeipuuv )
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// author           xeipuuv
// author-github    https://github.com/xeipuuv
// author-mail      xeipuuv@gmail.com
//
// repository-name  gojsonschema
// repository-desc  An implementation of JSON Schema, based on IETF's draft v4 - Go language.
//
// description      Contains const string and messages.
//
// created          01-01-2015

package gojsonschema

type (
	// locale is an interface for defining custom error strings
	locale interface {

		// False returns a format-string for "false" schema validation errors
		False() string

		// Required returns a format-string for "required" schema validation errors
		Required() string

		// InvalidType returns a format-string for "invalid type" schema validation errors
		InvalidType() string

		// NumberAnyOf returns a format-string for "anyOf" schema validation errors
		NumberAnyOf() string

		// NumberOneOf returns a format-string for "oneOf" schema validation errors
		NumberOneOf() string

		// NumberAllOf returns a format-string for "allOf" schema validation errors
		NumberAllOf() string

		// NumberNot returns a format-string to format a NumberNotError
		NumberNot() string

		// MissingDependency returns a format-string for "missing dependency" schema validation errors
		MissingDependency() string

		// Internal returns a format-string for internal errors
		Internal() string

		// Const returns a format-string to format a ConstError
		Const() string

		// Enum returns a format-string to format an EnumError
		Enum() string

		// ArrayNotEnoughItems returns a format-string to format an error for arrays having not enough items to match positional list of schema
		ArrayNotEnoughItems() string

		// ArrayNoAdditionalItems returns a format-string to format an ArrayNoAdditionalItemsError
		ArrayNoAdditionalItems() string

		// ArrayMinItems returns a format-string to format an ArrayMinItemsError
		ArrayMinItems() string

		// ArrayMaxItems returns a format-string to format an ArrayMaxItemsError
		ArrayMaxItems() string

		// Unique returns a format-string  to format an ItemsMustBeUniqueError
		Unique() string

		// ArrayContains returns a format-string to format an ArrayContainsError
		ArrayContains() string

		// ArrayMinProperties returns a format-string to format an ArrayMinPropertiesError
		ArrayMinProperties() string

		// ArrayMaxProperties returns a format-string to format an ArrayMaxPropertiesError
		ArrayMaxProperties() string

		// AdditionalPropertyNotAllowed returns a format-string to format an AdditionalPropertyNotAllowedError
		AdditionalPropertyNotAllowed() string

		// InvalidPropertyPattern returns a format-string to format an InvalidPropertyPatternError
		InvalidPropertyPattern() string

		// InvalidPropertyName returns a format-string to format an InvalidPropertyNameError
		InvalidPropertyName() string

		// StringGTE returns a format-string to format an StringLengthGTEError
		StringGTE() string

		// StringLTE returns a format-string to format an StringLengthLTEError
		StringLTE() string

		// DoesNotMatchPattern returns a format-string to format an DoesNotMatchPatternError
		DoesNotMatchPattern() string

		// DoesNotMatchFormat returns a format-string to format an DoesNotMatchFormatError
		DoesNotMatchFormat() string

		// MultipleOf returns a format-string to format an MultipleOfError
		MultipleOf() string

		// NumberGTE returns a format-string to format an NumberGTEError
		NumberGTE() string

		// NumberGT returns a format-string to format an NumberGTError
		NumberGT() string

		// NumberLTE returns a format-string to format an NumberLTEError
		NumberLTE() string

		// NumberLT returns a format-string to format an NumberLTError
		NumberLT() string

		// Schema validations

		// RegexPattern returns a format-string to format a regex-pattern error
		RegexPattern() string

		// GreaterThanZero returns a format-string to format an error where a number must be greater than zero
		GreaterThanZero() string

		// MustBeOfA returns a format-string to format an error where a value is of the wrong type
		MustBeOfA() string

		// MustBeOfAn returns a format-string to format an error where a value is of the wrong type
		MustBeOfAn() string

		// CannotBeUsedWithout returns a format-string to format a "cannot be used without" error
		CannotBeUsedWithout() string

		// CannotBeGT returns a format-string to format an error where a value are greater than allowed
		CannotBeGT() string

		// MustBeOfType returns a format-string to format an error where a value does not match the required type
		MustBeOfType() string

		// MustBeValidRegex returns a format-string to format an error where a regex is invalid
		MustBeValidRegex() string

		// MustBeValidFormat returns a format-string to format an error where a value does not match the expected format
		MustBeValidFormat() string

		// MustBeGTEZero returns a format-string to format an error where a value must be greater or equal than 0
		MustBeGTEZero() string

		// KeyCannotBeGreaterThan returns a format-string to format an error where a key is greater than the maximum  allowed
		KeyCannotBeGreaterThan() string

		// KeyItemsMustBeOfType returns a format-string to format an error where a key is of the wrong type
		KeyItemsMustBeOfType() string

		// KeyItemsMustBeUnique returns a format-string to format an error where keys are not unique
		KeyItemsMustBeUnique() string

		// ReferenceMustBeCanonical returns a format-string to format a "reference must be canonical" error
		ReferenceMustBeCanonical() string

		// NotAValidType returns a format-string to format an invalid type error
		NotAValidType() string

		// Duplicated returns a format-string to format an error where types are duplicated
		Duplicated() string

		// HttpBadStatus returns a format-string for errors when loading a schema using HTTP
		HttpBadStatus() string

		// ParseError returns a format-string for JSON parsing errors
		ParseError() string

		// ConditionThen returns a format-string for ConditionThenError errors
		ConditionThen() string

		// ConditionElse returns a format-string for ConditionElseError errors
		ConditionElse() string

		// ErrorFormat returns a format string for errors
		ErrorFormat() string
	}

	// DefaultLocale is the default locale for this package
	DefaultLocale struct{}
)

// False returns a format-string for "false" schema validation errors
func (l DefaultLocale) False() string {
	return "False always fails validation"
}

// Required returns a format-string for "required" schema validation errors
func (l DefaultLocale) Required() string {
	return `{{.property}} is required`
}

// InvalidType returns a format-string for "invalid type" schema validation errors
func (l DefaultLocale) InvalidType() string {
	return `Invalid type. Expected: {{.expected}}, given: {{.given}}`
}

// NumberAnyOf returns a format-string for "anyOf" schema validation errors
func (l DefaultLocale) NumberAnyOf() string {
	return `Must validate at least one schema (anyOf)`
}

// NumberOneOf returns a format-string for "oneOf" schema validation errors
func (l DefaultLocale) NumberOneOf() string {
	return `Must validate one and only one schema (oneOf)`
}

// NumberAllOf returns a format-string for "allOf" schema validation errors
func (l DefaultLocale) NumberAllOf() string {
	return `Must validate all the schemas (allOf)`
}

// NumberNot returns a format-string to format a NumberNotError
func (l DefaultLocale) NumberNot() string {
	return `Must not validate the schema (not)`
}

// MissingDependency returns a format-string for "missing dependency" schema validation errors
func (l DefaultLocale) MissingDependency() string {
	return `Has a dependency on {{.dependency}}`
}

// Internal returns a format-string for internal errors
func (l DefaultLocale) Internal() string {
	return `Internal Error {{.error}}`
}

// Const returns a format-string to format a ConstError
func (l DefaultLocale) Const() string {
	return `{{.field}} does not match: {{.allowed}}`
}

// Enum returns a format-string to format an EnumError
func (l DefaultLocale) Enum() string {
	return `{{.field}} must be one of the following: {{.allowed}}`
}

// ArrayNoAdditionalItems returns a format-string to format an ArrayNoAdditionalItemsError
func (l DefaultLocale) ArrayNoAdditionalItems() string {
	return `No additional items allowed on array`
}

// ArrayNotEnoughItems returns a format-string to format an error for arrays having not enough items to match positional list of schema
func (l DefaultLocale) ArrayNotEnoughItems() string {
	return `Not enough items on array to match positional list of schema`
}

// ArrayMinItems returns a format-string to format an ArrayMinItemsError
func (l DefaultLocale) ArrayMinItems() string {
	return `Array must have at least {{.min}} items`
}

// ArrayMaxItems returns a format-string to format an ArrayMaxItemsError
func (l DefaultLocale) ArrayMaxItems() string {
	return `Array must have at most {{.max}} items`
}

// Unique returns a format-string  to format an ItemsMustBeUniqueError
func (l DefaultLocale) Unique() string {
	return `{{.type}} items[{{.i}},{{.j}}] must be unique`
}

// ArrayContains returns a format-string to format an ArrayContainsError
func (l DefaultLocale) ArrayContains() string {
	return `At least one of the items must match`
}

// ArrayMinProperties returns a format-string to format an ArrayMinPropertiesError
func (l DefaultLocale) ArrayMinProperties() string {
	return `Must have at least {{.min}} properties`
}

// ArrayMaxProperties returns a format-string to format an ArrayMaxPropertiesError
func (l DefaultLocale) ArrayMaxProperties() string {
	return `Must have at most {{.max}} properties`
}

// AdditionalPropertyNotAllowed returns a format-string to format an AdditionalPropertyNotAllowedError
func (l DefaultLocale) AdditionalPropertyNotAllowed() string {
	return `Additional property {{.property}} is not allowed`
}

// InvalidPropertyPattern returns a format-string to format an InvalidPropertyPatternError
func (l DefaultLocale) InvalidPropertyPattern() string {
	return `Property "{{.property}}" does not match pattern {{.pattern}}`
}

// InvalidPropertyName returns a format-string to format an InvalidPropertyNameError
func (l DefaultLocale) InvalidPropertyName() string {
	return `Property name of "{{.property}}" does not match`
}

// StringGTE returns a format-string to format an StringLengthGTEError
func (l DefaultLocale) StringGTE() string {
	return `String length must be greater than or equal to {{.min}}`
}

// StringLTE returns a format-string to format an StringLengthLTEError
func (l DefaultLocale) StringLTE() string {
	return `String length must be less than or equal to {{.max}}`
}

// DoesNotMatchPattern returns a format-string to format an DoesNotMatchPatternError
func (l DefaultLocale) DoesNotMatchPattern() string {
	return `Does not match pattern '{{.pattern}}'`
}

// DoesNotMatchFormat returns a format-string to format an DoesNotMatchFormatError
func (l DefaultLocale) DoesNotMatchFormat() string {
	return `Does not match format '{{.format}}'`
}

// MultipleOf returns a format-string to format an MultipleOfError
func (l DefaultLocale) MultipleOf() string {
	return `Must be a multiple of {{.multiple}}`
}

// NumberGTE returns the format string to format a NumberGTEError
func (l DefaultLocale) NumberGTE() string {
	return `Must be greater than or equal to {{.min}}`
}

// NumberGT returns the format string to format a NumberGTError
func (l DefaultLocale) NumberGT() string {
	return `Must be greater than {{.min}}`
}

// NumberLTE returns the format string to format a NumberLTEError
func (l DefaultLocale) NumberLTE() string {
	return `Must be less than or equal to {{.max}}`
}

// NumberLT returns the format string to format a NumberLTError
func (l DefaultLocale) NumberLT() string {
	return `Must be less than {{.max}}`
}

// Schema validators

// RegexPattern returns a format-string to format a regex-pattern error
func (l DefaultLocale) RegexPattern() string {
	return `Invalid regex pattern '{{.pattern}}'`
}

// GreaterThanZero returns a format-string to format an error where a number must be greater than zero
func (l DefaultLocale) GreaterThanZero() string {
	return `{{.number}} must be strictly greater than 0`
}

// MustBeOfA returns a format-string to format an error where a value is of the wrong type
func (l DefaultLocale) MustBeOfA() string {
	return `{{.x}} must be of a {{.y}}`
}

// MustBeOfAn returns a format-string to format an error where a value is of the wrong type
func (l DefaultLocale) MustBeOfAn() string {
	return `{{.x}} must be of an {{.y}}`
}

// CannotBeUsedWithout returns a format-string to format a "cannot be used without" error
func (l DefaultLocale) CannotBeUsedWithout() string {
	return `{{.x}} cannot be used without {{.y}}`
}

// CannotBeGT returns a format-string to format an error where a value are greater than allowed
func (l DefaultLocale) CannotBeGT() string {
	return `{{.x}} cannot be greater than {{.y}}`
}

// MustBeOfType returns a format-string to format an error where a value does not match the required type
func (l DefaultLocale) MustBeOfType() string {
	return `{{.key}} must be of type {{.type}}`
}

// MustBeValidRegex returns a format-string to format an error where a regex is invalid
func (l DefaultLocale) MustBeValidRegex() string {
	return `{{.key}} must be a valid regex`
}

// MustBeValidFormat returns a format-string to format an error where a value does not match the expected format
func (l DefaultLocale) MustBeValidFormat() string {
	return `{{.key}} must be a valid format {{.given}}`
}

// MustBeGTEZero returns a format-string to format an error where a value must be greater or equal than 0
func (l DefaultLocale) MustBeGTEZero() string {
	return `{{.key}} must be greater than or equal to 0`
}

// KeyCannotBeGreaterThan returns a format-string to format an error where a value is greater than the maximum  allowed
func (l DefaultLocale) KeyCannotBeGreaterThan() string {
	return `{{.key}} cannot be greater than {{.y}}`
}

// KeyItemsMustBeOfType returns a format-string to format an error where a key is of the wrong type
func (l DefaultLocale) KeyItemsMustBeOfType() string {
	return `{{.key}} items must be {{.type}}`
}

// KeyItemsMustBeUnique returns a format-string to format an error where keys are not unique
func (l DefaultLocale) KeyItemsMustBeUnique() string {
	return `{{.key}} items must be unique`
}

// ReferenceMustBeCanonical returns a format-string to format a "reference must be canonical" error
func (l DefaultLocale) ReferenceMustBeCanonical() string {
	return `Reference {{.reference}} must be canonical`
}

// NotAValidType returns a format-string to format an invalid type error
func (l DefaultLocale) NotAValidType() string {
	return `has a primitive type that is NOT VALID -- given: {{.given}} Expected valid values are:{{.expected}}`
}

// Duplicated returns a format-string to format an error where types are duplicated
func (l DefaultLocale) Duplicated() string {
	return `{{.type}} type is duplicated`
}

// HttpBadStatus returns a format-string for errors when loading a schema using HTTP
func (l DefaultLocale) HttpBadStatus() string {
	return `Could not read schema from HTTP, response status is {{.status}}`
}

// ErrorFormat returns a format string for errors
// Replacement options: field, description, context, value
func (l DefaultLocale) ErrorFormat() string {
	return `{{.field}}: {{.description}}`
}

// ParseError returns a format-string for JSON parsing errors
func (l DefaultLocale) ParseError() string {
	return `Expected: {{.expected}}, given: Invalid JSON`
}

// ConditionThen returns a format-string for ConditionThenError errors
// If/Else
func (l DefaultLocale) ConditionThen() string {
	return `Must validate "then" as "if" was valid`
}

// ConditionElse returns a format-string for ConditionElseError errors
func (l DefaultLocale) ConditionElse() string {
	return `Must validate "else" as "if" was not valid`
}

// constants
const (
	STRING_NUMBER                     = "number"
	STRING_ARRAY_OF_STRINGS           = "array of strings"
	STRING_ARRAY_OF_SCHEMAS           = "array of schemas"
	STRING_SCHEMA                     = "valid schema"
	STRING_SCHEMA_OR_ARRAY_OF_STRINGS = "schema or array of strings"
	STRING_PROPERTIES                 = "properties"
	STRING_DEPENDENCY                 = "dependency"
	STRING_PROPERTY                   = "property"
	STRING_UNDEFINED                  = "undefined"
	STRING_CONTEXT_ROOT               = "(root)"
	STRING_ROOT_SCHEMA_PROPERTY       = "(root)"
)
