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
	"fmt"
	"reflect"

	"k8s.io/kube-openapi/pkg/validation/spec"
	"k8s.io/kube-openapi/pkg/validation/strfmt"
)

type schemaPropsValidator struct {
	Path            string
	In              string
	AllOf           []spec.Schema
	OneOf           []spec.Schema
	AnyOf           []spec.Schema
	Not             *spec.Schema
	Dependencies    spec.Dependencies
	anyOfValidators []SchemaValidator
	allOfValidators []SchemaValidator
	oneOfValidators []SchemaValidator
	notValidator    *SchemaValidator
	Root            interface{}
	KnownFormats    strfmt.Registry
	Options         SchemaValidatorOptions
}

func (s *schemaPropsValidator) SetPath(path string) {
	s.Path = path
	for _, v := range s.anyOfValidators {
		v.SetPath(path)
	}
	for _, v := range s.allOfValidators {
		v.SetPath(path)
	}
	for _, v := range s.oneOfValidators {
		v.SetPath(path)
	}
	if s.notValidator != nil {
		s.notValidator.SetPath(path)
	}
}

func newSchemaPropsValidator(path string, in string, allOf, oneOf, anyOf []spec.Schema, not *spec.Schema, deps spec.Dependencies, root interface{}, formats strfmt.Registry, options ...Option) *schemaPropsValidator {
	var anyValidators []SchemaValidator
	for _, v := range anyOf {
		v := v
		anyValidators = append(anyValidators, *NewSchemaValidator(&v, root, path, formats, options...))
	}
	var allValidators []SchemaValidator
	for _, v := range allOf {
		v := v
		allValidators = append(allValidators, *NewSchemaValidator(&v, root, path, formats, options...))
	}
	var oneValidators []SchemaValidator
	for _, v := range oneOf {
		v := v
		oneValidators = append(oneValidators, *NewSchemaValidator(&v, root, path, formats, options...))
	}

	var notValidator *SchemaValidator
	if not != nil {
		notValidator = NewSchemaValidator(not, root, path, formats, options...)
	}

	schOptions := &SchemaValidatorOptions{}
	for _, o := range options {
		o(schOptions)
	}
	return &schemaPropsValidator{
		Path:            path,
		In:              in,
		AllOf:           allOf,
		OneOf:           oneOf,
		AnyOf:           anyOf,
		Not:             not,
		Dependencies:    deps,
		anyOfValidators: anyValidators,
		allOfValidators: allValidators,
		oneOfValidators: oneValidators,
		notValidator:    notValidator,
		Root:            root,
		KnownFormats:    formats,
		Options:         *schOptions,
	}
}

func (s *schemaPropsValidator) Applies(source interface{}, kind reflect.Kind) bool {
	r := reflect.TypeOf(source) == specSchemaType
	debugLog("schema props validator for %q applies %t for %T (kind: %v)\n", s.Path, r, source, kind)
	return r
}

func (s *schemaPropsValidator) Validate(data interface{}) *Result {
	mainResult := new(Result)

	// Intermediary error results

	// IMPORTANT! messages from underlying validators
	keepResultAnyOf := new(Result)
	keepResultOneOf := new(Result)
	keepResultAllOf := new(Result)

	// Validates at least one in anyOf schemas
	var firstSuccess *Result
	if len(s.anyOfValidators) > 0 {
		var bestFailures *Result
		succeededOnce := false
		for _, anyOfSchema := range s.anyOfValidators {
			result := anyOfSchema.Validate(data)
			// We keep inner IMPORTANT! errors no matter what MatchCount tells us
			keepResultAnyOf.Merge(result.keepRelevantErrors())
			if result.IsValid() {
				bestFailures = nil
				succeededOnce = true
				if firstSuccess == nil {
					firstSuccess = result
				}
				keepResultAnyOf = new(Result)
				break
			}
			// MatchCount is used to select errors from the schema with most positive checks
			if bestFailures == nil || result.MatchCount > bestFailures.MatchCount {
				bestFailures = result
			}
		}

		if !succeededOnce {
			mainResult.AddErrors(mustValidateAtLeastOneSchemaMsg(s.Path))
		}
		if bestFailures != nil {
			mainResult.Merge(bestFailures)
		} else if firstSuccess != nil {
			mainResult.Merge(firstSuccess)
		}
	}

	// Validates exactly one in oneOf schemas
	if len(s.oneOfValidators) > 0 {
		var bestFailures *Result
		var firstSuccess *Result
		validated := 0

		for _, oneOfSchema := range s.oneOfValidators {
			result := oneOfSchema.Validate(data)
			// We keep inner IMPORTANT! errors no matter what MatchCount tells us
			keepResultOneOf.Merge(result.keepRelevantErrors())
			if result.IsValid() {
				validated++
				bestFailures = nil
				if firstSuccess == nil {
					firstSuccess = result
				}
				keepResultOneOf = new(Result)
				continue
			}
			// MatchCount is used to select errors from the schema with most positive checks
			if validated == 0 && (bestFailures == nil || result.MatchCount > bestFailures.MatchCount) {
				bestFailures = result
			}
		}

		if validated != 1 {
			additionalMsg := ""
			if validated == 0 {
				additionalMsg = "Found none valid"
			} else {
				additionalMsg = fmt.Sprintf("Found %d valid alternatives", validated)
			}

			mainResult.AddErrors(mustValidateOnlyOneSchemaMsg(s.Path, additionalMsg))
			if bestFailures != nil {
				mainResult.Merge(bestFailures)
			}
		} else if firstSuccess != nil {
			mainResult.Merge(firstSuccess)
		}
	}

	// Validates all of allOf schemas
	if len(s.allOfValidators) > 0 {
		validated := 0

		for _, allOfSchema := range s.allOfValidators {
			result := allOfSchema.Validate(data)
			// We keep inner IMPORTANT! errors no matter what MatchCount tells us
			keepResultAllOf.Merge(result.keepRelevantErrors())
			//keepResultAllOf.Merge(result)
			if result.IsValid() {
				validated++
			}
			mainResult.Merge(result)
		}

		if validated != len(s.allOfValidators) {
			additionalMsg := ""
			if validated == 0 {
				additionalMsg = ". None validated"
			}

			mainResult.AddErrors(mustValidateAllSchemasMsg(s.Path, additionalMsg))
		}
	}

	if s.notValidator != nil {
		result := s.notValidator.Validate(data)
		// We keep inner IMPORTANT! errors no matter what MatchCount tells us
		if result.IsValid() {
			mainResult.AddErrors(mustNotValidatechemaMsg(s.Path))
		}
	}

	if s.Dependencies != nil && len(s.Dependencies) > 0 && reflect.TypeOf(data).Kind() == reflect.Map {
		val := data.(map[string]interface{})
		for key := range val {
			if dep, ok := s.Dependencies[key]; ok {

				if dep.Schema != nil {
					mainResult.Merge(NewSchemaValidator(dep.Schema, s.Root, s.Path+"."+key, s.KnownFormats, s.Options.Options()...).Validate(data))
					continue
				}

				if len(dep.Property) > 0 {
					for _, depKey := range dep.Property {
						if _, ok := val[depKey]; !ok {
							mainResult.AddErrors(hasADependencyMsg(s.Path, depKey))
						}
					}
				}
			}
		}
	}

	mainResult.Inc()
	// In the end we retain best failures for schema validation
	// plus, if any, composite errors which may explain special cases (tagged as IMPORTANT!).
	return mainResult.Merge(keepResultAllOf, keepResultOneOf, keepResultAnyOf)
}
