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
	"k8s.io/kube-openapi/pkg/validation/errors"
)

// Error messages related to schema validation and returned as results.
const (
	// ArrayDoesNotAllowAdditionalItemsError when an additionalItems construct is not verified by the array values provided.
	//
	// TODO: should move to package go-openapi/errors
	ArrayDoesNotAllowAdditionalItemsError = "array doesn't allow for additional items"

	// HasDependencyError indicates that a dependencies construct was not verified
	HasDependencyError = "%q has a dependency on %s"

	// InvalidTypeConversionError indicates that a numerical conversion for the given type could not be carried on
	InvalidTypeConversionError = "invalid type conversion in %s: %v "

	// MustValidateAtLeastOneSchemaError indicates that in a AnyOf construct, none of the schema constraints specified were verified
	MustValidateAtLeastOneSchemaError = "%q must validate at least one schema (anyOf)"

	// MustValidateOnlyOneSchemaError indicates that in a OneOf construct, either none of the schema constraints specified were verified, or several were
	MustValidateOnlyOneSchemaError = "%q must validate one and only one schema (oneOf). %s"

	// MustValidateAllSchemasError indicates that in a AllOf construct, at least one of the schema constraints specified were not verified
	//
	// TODO: punctuation in message
	MustValidateAllSchemasError = "%q must validate all the schemas (allOf)%s"

	// MustNotValidateSchemaError indicates that in a Not construct, the schema constraint specified was verified
	MustNotValidateSchemaError = "%q must not validate the schema (not)"
)

// Warning messages related to schema validation and returned as results
const ()

func invalidTypeConversionMsg(path string, err error) errors.Error {
	return errors.New(errors.CompositeErrorCode, InvalidTypeConversionError, path, err)
}
func mustValidateOnlyOneSchemaMsg(path, additionalMsg string) errors.Error {
	return errors.New(errors.CompositeErrorCode, MustValidateOnlyOneSchemaError, path, additionalMsg)
}
func mustValidateAtLeastOneSchemaMsg(path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, MustValidateAtLeastOneSchemaError, path)
}
func mustValidateAllSchemasMsg(path, additionalMsg string) errors.Error {
	return errors.New(errors.CompositeErrorCode, MustValidateAllSchemasError, path, additionalMsg)
}
func mustNotValidatechemaMsg(path string) errors.Error {
	return errors.New(errors.CompositeErrorCode, MustNotValidateSchemaError, path)
}
func hasADependencyMsg(path, depkey string) errors.Error {
	return errors.New(errors.CompositeErrorCode, HasDependencyError, path, depkey)
}
func arrayDoesNotAllowAdditionalItemsMsg() errors.Error {
	return errors.New(errors.CompositeErrorCode, ArrayDoesNotAllowAdditionalItemsError)
}
