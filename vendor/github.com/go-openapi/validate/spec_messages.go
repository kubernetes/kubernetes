// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package validate

import (
	"net/http"

	"github.com/go-openapi/errors"
)

// Error messages related to spec validation and returned as results.
const (
	// ArrayRequiresItemsError ...
	ArrayRequiresItemsError = "%s for %q is a collection without an element type (array requires items definition)"

	// ArrayInParamRequiresItemsError ...
	ArrayInParamRequiresItemsError = "param %q for %q is a collection without an element type (array requires item definition)"

	// ArrayInHeaderRequiresItemsError ...
	ArrayInHeaderRequiresItemsError = "header %q for %q is a collection without an element type (array requires items definition)"

	// BothFormDataAndBodyError indicates that an operation specifies both a body and a formData parameter, which is forbidden
	BothFormDataAndBodyError = "operation %q has both formData and body parameters. Only one such In: type may be used for a given operation"

	// CannotResolveRefError when a $ref could not be resolved
	CannotResolveReferenceError = "could not resolve reference in %s to $ref %s: %v"

	// CircularAncestryDefinitionError ...
	CircularAncestryDefinitionError = "definition %q has circular ancestry: %v"

	// DefaultValueDoesNotValidateError results from an invalid default value provided
	DefaultValueDoesNotValidateError = "default value for %s in %s does not validate its schema"

	// DefaultValueItemsDoesNotValidateError results from an invalid default value provided for Items
	DefaultValueItemsDoesNotValidateError = "default value for %s.items in %s does not validate its schema"

	// DefaultValueHeaderDoesNotValidateError results from an invalid default value provided in header
	DefaultValueHeaderDoesNotValidateError = "in operation %q, default value in header %s for %s does not validate its schema"

	// DefaultValueHeaderItemsDoesNotValidateError results from an invalid default value provided in header.items
	DefaultValueHeaderItemsDoesNotValidateError = "in operation %q, default value in header.items %s for %s does not validate its schema"

	// DefaultValueInDoesNotValidateError ...
	DefaultValueInDoesNotValidateError = "in operation %q, default value in %s does not validate its schema"

	// DuplicateParamNameError ...
	DuplicateParamNameError = "duplicate parameter name %q for %q in operation %q"

	// DuplicatePropertiesError ...
	DuplicatePropertiesError = "definition %q contains duplicate properties: %v"

	// ExampleValueDoesNotValidateError results from an invalid example value provided
	ExampleValueDoesNotValidateError = "example value for %s in %s does not validate its schema"

	// ExampleValueItemsDoesNotValidateError results from an invalid example value provided for Items
	ExampleValueItemsDoesNotValidateError = "example value for %s.items in %s does not validate its schema"

	// ExampleValueHeaderDoesNotValidateError results from an invalid example value provided in header
	ExampleValueHeaderDoesNotValidateError = "in operation %q, example value in header %s for %s does not validate its schema"

	// ExampleValueHeaderItemsDoesNotValidateError results from an invalid example value provided in header.items
	ExampleValueHeaderItemsDoesNotValidateError = "in operation %q, example value in header.items %s for %s does not validate its schema"

	// ExampleValueInDoesNotValidateError ...
	ExampleValueInDoesNotValidateError = "in operation %q, example value in %s does not validate its schema"

	// EmptyPathParameterError means that a path parameter was found empty (e.g. "{}")
	EmptyPathParameterError = "%q contains an empty path parameter"

	// InvalidDocumentError states that spec validation only processes spec.Document objects
	InvalidDocumentError = "spec validator can only validate spec.Document objects"

	// InvalidItemsPatternError indicates an Items definition with invalid pattern
	InvalidItemsPatternError = "%s for %q has invalid items pattern: %q"

	// InvalidParameterDefinitionError indicates an error detected on a parameter definition
	InvalidParameterDefinitionError = "invalid definition for parameter %s in %s in operation %q"

	// InvalidParameterDefinitionAsSchemaError indicates an error detected on a parameter definition, which was mistaken with a schema definition.
	// Most likely, this situation is encountered whenever a $ref has been added as a sibling of the parameter definition.
	InvalidParameterDefinitionAsSchemaError = "invalid definition as Schema for parameter %s in %s in operation %q"

	// InvalidPatternError ...
	InvalidPatternError = "pattern %q is invalid in %s"

	// InvalidPatternInError indicates an invalid pattern in a schema or items definition
	InvalidPatternInError = "%s in %s has invalid pattern: %q"

	// InvalidPatternInHeaderError indicates a header definition with an invalid pattern
	InvalidPatternInHeaderError = "in operation %q, header %s for %s has invalid pattern %q: %v"

	// InvalidPatternInParamError ...
	InvalidPatternInParamError = "operation %q has invalid pattern in param %q: %q"

	// InvalidReferenceError indicates that a $ref property could not be resolved
	InvalidReferenceError = "invalid ref %q"

	// InvalidResponseDefinitionAsSchemaError indicates an error detected on a response definition, which was mistaken with a schema definition.
	// Most likely, this situation is encountered whenever a $ref has been added as a sibling of the response definition.
	InvalidResponseDefinitionAsSchemaError = "invalid definition as Schema for response %s in %s"

	// MultipleBodyParamError indicates that an operation specifies multiple parameter with in: body
	MultipleBodyParamError = "operation %q has more than 1 body param: %v"

	// NonUniqueOperationIDError indicates that the same operationId has been specified several times
	NonUniqueOperationIDError = "%q is defined %d times"

	// NoParameterInPathError indicates that a path was found without any parameter
	NoParameterInPathError = "path param %q has no parameter definition"

	// NoValidPathErrorOrWarning indicates that no single path could be validated. If Paths is empty, this message is only a warning.
	NoValidPathErrorOrWarning = "spec has no valid path defined"

	// NoValidResponseError indicates that no valid response description could be found for an operation
	NoValidResponseError = "operation %q has no valid response"

	// PathOverlapError ...
	PathOverlapError = "path %s overlaps with %s"

	// PathParamNotInPathError indicates that a parameter specified with in: path was not found in the path specification
	PathParamNotInPathError = "path param %q is not present in path %q"

	// PathParamNotUniqueError ...
	PathParamNotUniqueError = "params in path %q must be unique: %q conflicts with %q"

	// PathParamNotRequiredError ...
	PathParamRequiredError = "in operation %q,path param %q must be declared as required"

	// RefNotAllowedInHeaderError indicates a $ref was found in a header definition, which is not allowed by Swagger
	RefNotAllowedInHeaderError = "IMPORTANT!in %q: $ref are not allowed in headers. In context for header %q%s"

	// RequiredButNotDefinedError ...
	RequiredButNotDefinedError = "%q is present in required but not defined as property in definition %q"

	// SomeParametersBrokenError indicates that some parameters could not be resolved, which might result in partial checks to be carried on
	SomeParametersBrokenError = "some parameters definitions are broken in %q.%s. Cannot carry on full checks on parameters for operation %s"

	// UnresolvedReferencesError indicates that at least one $ref could not be resolved
	UnresolvedReferencesError = "some references could not be resolved in spec. First found: %v"
)

// Warning messages related to spec validation and returned as results
const (
	// ExamplesWithoutSchemaWarning indicates that examples are provided for a response,but not schema to validate the example against
	ExamplesWithoutSchemaWarning = "Examples provided without schema in operation %q, %s"

	// ExamplesMimeNotSupportedWarning indicates that examples are provided with a mime type different than application/json, which
	// the validator dos not support yetl
	ExamplesMimeNotSupportedWarning = "No validation attempt for examples for media types other than application/json, in operation %q, %s"

	// PathParamGarbledWarning ...
	PathParamGarbledWarning = "in path %q, param %q contains {,} or white space. Albeit not stricly illegal, this is probably no what you want"

	// PathStrippedParamGarbledWarning ...
	PathStrippedParamGarbledWarning = "path stripped from path parameters %s contains {,} or white space. This is probably no what you want."

	// ReadOnlyAndRequiredWarning ...
	ReadOnlyAndRequiredWarning = "Required property %s in %q should not be marked as both required and readOnly"

	// RefShouldNotHaveSiblingsWarning indicates that a $ref was found with a sibling definition. This results in the $ref taking over its siblings,
	// which is most likely not wanted.
	RefShouldNotHaveSiblingsWarning = "$ref property should have no sibling in %q.%s"

	// RequiredHasDefaultWarning indicates that a required parameter property should not have a default
	RequiredHasDefaultWarning = "%s in %s has a default value and is required as parameter"

	// UnusedDefinitionWarning ...
	UnusedDefinitionWarning = "definition %q is not used anywhere"

	// UnusedParamWarning ...
	UnusedParamWarning = "parameter %q is not used anywhere"

	// UnusedResponseWarning ...
	UnusedResponseWarning = "response %q is not used anywhere"
)

// Additional error codes
const (
	// InternalErrorCode reports an internal technical error
	InternalErrorCode = http.StatusInternalServerError
	// NotFoundErrorCode indicates that a resource (e.g. a $ref) could not be found
	NotFoundErrorCode = http.StatusNotFound
)

func invalidDocumentMsg() errors.Error {
	return errors.New(InternalErrorCode, InvalidDocumentError)
}
func invalidRefMsg(path string) errors.Error {
	return errors.New(NotFoundErrorCode, InvalidReferenceError, path)
}
func unresolvedReferencesMsg(err error) errors.Error {
	return errors.New(errors.CompositeErrorCode, UnresolvedReferencesError, err)
}
func noValidPathMsg() errors.Error {
	return errors.New(errors.CompositeErrorCode, NoValidPathErrorOrWarning)
}
func emptyPathParameterMsg(path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, EmptyPathParameterError, path)
}
func nonUniqueOperationIDMsg(path string, i int) errors.Error {
	return errors.New(errors.CompositeErrorCode, NonUniqueOperationIDError, path, i)
}
func circularAncestryDefinitionMsg(path string, args interface{}) errors.Error {
	return errors.New(errors.CompositeErrorCode, CircularAncestryDefinitionError, path, args)
}
func duplicatePropertiesMsg(path string, args interface{}) errors.Error {
	return errors.New(errors.CompositeErrorCode, DuplicatePropertiesError, path, args)
}
func pathParamNotInPathMsg(path, param string) errors.Error {
	return errors.New(errors.CompositeErrorCode, PathParamNotInPathError, param, path)
}
func arrayRequiresItemsMsg(path, operation string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ArrayRequiresItemsError, path, operation)
}
func arrayInParamRequiresItemsMsg(path, operation string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ArrayInParamRequiresItemsError, path, operation)
}
func arrayInHeaderRequiresItemsMsg(path, operation string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ArrayInHeaderRequiresItemsError, path, operation)
}
func invalidItemsPatternMsg(path, operation, pattern string) errors.Error {
	return errors.New(errors.CompositeErrorCode, InvalidItemsPatternError, path, operation, pattern)
}
func invalidPatternMsg(pattern, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, InvalidPatternError, pattern, path)
}
func requiredButNotDefinedMsg(path, definition string) errors.Error {
	return errors.New(errors.CompositeErrorCode, RequiredButNotDefinedError, path, definition)
}
func pathParamGarbledMsg(path, param string) errors.Error {
	return errors.New(errors.CompositeErrorCode, PathParamGarbledWarning, path, param)
}
func pathStrippedParamGarbledMsg(path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, PathStrippedParamGarbledWarning, path)
}
func pathOverlapMsg(path, arg string) errors.Error {
	return errors.New(errors.CompositeErrorCode, PathOverlapError, path, arg)
}
func invalidPatternInParamMsg(operation, param, pattern string) errors.Error {
	return errors.New(errors.CompositeErrorCode, InvalidPatternInParamError, operation, param, pattern)
}
func pathParamRequiredMsg(operation, param string) errors.Error {
	return errors.New(errors.CompositeErrorCode, PathParamRequiredError, operation, param)
}
func bothFormDataAndBodyMsg(operation string) errors.Error {
	return errors.New(errors.CompositeErrorCode, BothFormDataAndBodyError, operation)
}
func multipleBodyParamMsg(operation string, args interface{}) errors.Error {
	return errors.New(errors.CompositeErrorCode, MultipleBodyParamError, operation, args)
}
func pathParamNotUniqueMsg(path, param, arg string) errors.Error {
	return errors.New(errors.CompositeErrorCode, PathParamNotUniqueError, path, param, arg)
}
func duplicateParamNameMsg(path, param, operation string) errors.Error {
	return errors.New(errors.CompositeErrorCode, DuplicateParamNameError, param, path, operation)
}
func unusedParamMsg(arg string) errors.Error {
	return errors.New(errors.CompositeErrorCode, UnusedParamWarning, arg)
}
func unusedDefinitionMsg(arg string) errors.Error {
	return errors.New(errors.CompositeErrorCode, UnusedDefinitionWarning, arg)
}
func unusedResponseMsg(arg string) errors.Error {
	return errors.New(errors.CompositeErrorCode, UnusedResponseWarning, arg)
}
func readOnlyAndRequiredMsg(path, param string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ReadOnlyAndRequiredWarning, param, path)
}
func noParameterInPathMsg(param string) errors.Error {
	return errors.New(errors.CompositeErrorCode, NoParameterInPathError, param)
}
func requiredHasDefaultMsg(param, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, RequiredHasDefaultWarning, param, path)
}
func defaultValueDoesNotValidateMsg(param, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, DefaultValueDoesNotValidateError, param, path)
}
func defaultValueItemsDoesNotValidateMsg(param, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, DefaultValueItemsDoesNotValidateError, param, path)
}
func noValidResponseMsg(operation string) errors.Error {
	return errors.New(errors.CompositeErrorCode, NoValidResponseError, operation)
}
func defaultValueHeaderDoesNotValidateMsg(operation, header, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, DefaultValueHeaderDoesNotValidateError, operation, header, path)
}
func defaultValueHeaderItemsDoesNotValidateMsg(operation, header, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, DefaultValueHeaderItemsDoesNotValidateError, operation, header, path)
}
func invalidPatternInHeaderMsg(operation, header, path, pattern string, args interface{}) errors.Error {
	return errors.New(errors.CompositeErrorCode, InvalidPatternInHeaderError, operation, header, path, pattern, args)
}
func invalidPatternInMsg(path, in, pattern string) errors.Error {
	return errors.New(errors.CompositeErrorCode, InvalidPatternInError, path, in, pattern)
}
func defaultValueInDoesNotValidateMsg(operation, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, DefaultValueInDoesNotValidateError, operation, path)
}
func exampleValueDoesNotValidateMsg(param, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ExampleValueDoesNotValidateError, param, path)
}
func exampleValueItemsDoesNotValidateMsg(param, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ExampleValueItemsDoesNotValidateError, param, path)
}
func exampleValueHeaderDoesNotValidateMsg(operation, header, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ExampleValueHeaderDoesNotValidateError, operation, header, path)
}
func exampleValueHeaderItemsDoesNotValidateMsg(operation, header, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ExampleValueHeaderItemsDoesNotValidateError, operation, header, path)
}
func exampleValueInDoesNotValidateMsg(operation, path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ExampleValueInDoesNotValidateError, operation, path)
}
func examplesWithoutSchemaMsg(operation, response string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ExamplesWithoutSchemaWarning, operation, response)
}
func examplesMimeNotSupportedMsg(operation, response string) errors.Error {
	return errors.New(errors.CompositeErrorCode, ExamplesMimeNotSupportedWarning, operation, response)
}
func refNotAllowedInHeaderMsg(path, header, ref string) errors.Error {
	return errors.New(errors.CompositeErrorCode, RefNotAllowedInHeaderError, path, header, ref)
}
func cannotResolveRefMsg(path, ref string, err error) errors.Error {
	return errors.New(errors.CompositeErrorCode, CannotResolveReferenceError, path, ref, err)
}
func invalidParameterDefinitionMsg(path, method, operationID string) errors.Error {
	return errors.New(errors.CompositeErrorCode, InvalidParameterDefinitionError, path, method, operationID)
}
func invalidParameterDefinitionAsSchemaMsg(path, method, operationID string) errors.Error {
	return errors.New(errors.CompositeErrorCode, InvalidParameterDefinitionAsSchemaError, path, method, operationID)
}

// disabled
//func invalidResponseDefinitionAsSchemaMsg(path, method string) errors.Error {
//	return errors.New(errors.CompositeErrorCode, InvalidResponseDefinitionAsSchemaError, path, method)
//}
func someParametersBrokenMsg(path, method, operationID string) errors.Error {
	return errors.New(errors.CompositeErrorCode, SomeParametersBrokenError, path, method, operationID)
}
func refShouldNotHaveSiblingsMsg(path, operationID string) errors.Error {
	return errors.New(errors.CompositeErrorCode, RefShouldNotHaveSiblingsWarning, operationID, path)
}
