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

	"github.com/go-openapi/spec"
)

// ExampleValidator validates example values defined in a spec
type exampleValidator struct {
	SpecValidator  *SpecValidator
	visitedSchemas map[string]bool
}

// resetVisited resets the internal state of visited schemas
func (ex *exampleValidator) resetVisited() {
	ex.visitedSchemas = map[string]bool{}
}

// beingVisited asserts a schema is being visited
func (ex *exampleValidator) beingVisited(path string) {
	ex.visitedSchemas[path] = true
}

// isVisited tells if a path has already been visited
func (ex *exampleValidator) isVisited(path string) bool {
	return isVisited(path, ex.visitedSchemas)
}

// Validate validates the example values declared in the swagger spec
// Example values MUST conform to their schema.
//
// With Swagger 2.0, examples are supported in:
//   - schemas
//   - individual property
//   - responses
//
func (ex *exampleValidator) Validate() (errs *Result) {
	errs = new(Result)
	if ex == nil || ex.SpecValidator == nil {
		return errs
	}
	ex.resetVisited()
	errs.Merge(ex.validateExampleValueValidAgainstSchema()) // error -

	return errs
}

func (ex *exampleValidator) validateExampleValueValidAgainstSchema() *Result {
	// every example value that is specified must validate against the schema for that property
	// in: schemas, properties, object, items
	// not in: headers, parameters without schema

	res := new(Result)
	s := ex.SpecValidator

	for method, pathItem := range s.analyzer.Operations() {
		for path, op := range pathItem {
			// parameters
			for _, param := range paramHelp.safeExpandedParamsFor(path, method, op.ID, res, s) {

				// As of swagger 2.0, Examples are not supported in simple parameters
				// However, it looks like it is supported by go-openapi

				// reset explored schemas to get depth-first recursive-proof exploration
				ex.resetVisited()

				// Check simple parameters first
				// default values provided must validate against their inline definition (no explicit schema)
				if param.Example != nil && param.Schema == nil {
					// check param default value is valid
					red := NewParamValidator(&param, s.KnownFormats).Validate(param.Example)
					if red.HasErrorsOrWarnings() {
						res.AddWarnings(exampleValueDoesNotValidateMsg(param.Name, param.In))
						res.MergeAsWarnings(red)
					}
				}

				// Recursively follows Items and Schemas
				if param.Items != nil {
					red := ex.validateExampleValueItemsAgainstSchema(param.Name, param.In, &param, param.Items)
					if red.HasErrorsOrWarnings() {
						res.AddWarnings(exampleValueItemsDoesNotValidateMsg(param.Name, param.In))
						res.Merge(red)
					}
				}

				if param.Schema != nil {
					// Validate example value against schema
					red := ex.validateExampleValueSchemaAgainstSchema(param.Name, param.In, param.Schema)
					if red.HasErrorsOrWarnings() {
						res.AddWarnings(exampleValueDoesNotValidateMsg(param.Name, param.In))
						res.Merge(red)
					}
				}
			}

			if op.Responses != nil {
				if op.Responses.Default != nil {
					// Same constraint on default Response
					res.Merge(ex.validateExampleInResponse(op.Responses.Default, jsonDefault, path, 0, op.ID))
				}
				// Same constraint on regular Responses
				if op.Responses.StatusCodeResponses != nil { // Safeguard
					for code, r := range op.Responses.StatusCodeResponses {
						res.Merge(ex.validateExampleInResponse(&r, "response", path, code, op.ID))
					}
				}
			} else if op.ID != "" {
				// Empty op.ID means there is no meaningful operation: no need to report a specific message
				res.AddErrors(noValidResponseMsg(op.ID))
			}
		}
	}
	if s.spec.Spec().Definitions != nil { // Safeguard
		// reset explored schemas to get depth-first recursive-proof exploration
		ex.resetVisited()
		for nm, sch := range s.spec.Spec().Definitions {
			res.Merge(ex.validateExampleValueSchemaAgainstSchema(fmt.Sprintf("definitions.%s", nm), "body", &sch))
		}
	}
	return res
}

func (ex *exampleValidator) validateExampleInResponse(resp *spec.Response, responseType, path string, responseCode int, operationID string) *Result {
	s := ex.SpecValidator

	response, res := responseHelp.expandResponseRef(resp, path, s)
	if !res.IsValid() { // Safeguard
		return res
	}

	responseName, responseCodeAsStr := responseHelp.responseMsgVariants(responseType, responseCode)

	// nolint: dupl
	if response.Headers != nil { // Safeguard
		for nm, h := range response.Headers {
			// reset explored schemas to get depth-first recursive-proof exploration
			ex.resetVisited()

			if h.Example != nil {
				red := NewHeaderValidator(nm, &h, s.KnownFormats).Validate(h.Example)
				if red.HasErrorsOrWarnings() {
					res.AddWarnings(exampleValueHeaderDoesNotValidateMsg(operationID, nm, responseName))
					res.MergeAsWarnings(red)
				}
			}

			// Headers have inline definition, like params
			if h.Items != nil {
				red := ex.validateExampleValueItemsAgainstSchema(nm, "header", &h, h.Items)
				if red.HasErrorsOrWarnings() {
					res.AddWarnings(exampleValueHeaderItemsDoesNotValidateMsg(operationID, nm, responseName))
					res.MergeAsWarnings(red)
				}
			}

			if _, err := compileRegexp(h.Pattern); err != nil {
				res.AddErrors(invalidPatternInHeaderMsg(operationID, nm, responseName, h.Pattern, err))
			}

			// Headers don't have schema
		}
	}
	if response.Schema != nil {
		// reset explored schemas to get depth-first recursive-proof exploration
		ex.resetVisited()

		red := ex.validateExampleValueSchemaAgainstSchema(responseCodeAsStr, "response", response.Schema)
		if red.HasErrorsOrWarnings() {
			// Additional message to make sure the context of the error is not lost
			res.AddWarnings(exampleValueInDoesNotValidateMsg(operationID, responseName))
			res.Merge(red)
		}
	}

	if response.Examples != nil {
		if response.Schema != nil {
			if example, ok := response.Examples["application/json"]; ok {
				res.MergeAsWarnings(NewSchemaValidator(response.Schema, s.spec.Spec(), path+".examples", s.KnownFormats, SwaggerSchema(true)).Validate(example))
			} else {
				// TODO: validate other media types too
				res.AddWarnings(examplesMimeNotSupportedMsg(operationID, responseName))
			}
		} else {
			res.AddWarnings(examplesWithoutSchemaMsg(operationID, responseName))
		}
	}
	return res
}

func (ex *exampleValidator) validateExampleValueSchemaAgainstSchema(path, in string, schema *spec.Schema) *Result {
	if schema == nil || ex.isVisited(path) {
		// Avoids recursing if we are already done with that check
		return nil
	}
	ex.beingVisited(path)
	s := ex.SpecValidator
	res := new(Result)

	if schema.Example != nil {
		res.MergeAsWarnings(NewSchemaValidator(schema, s.spec.Spec(), path+".example", s.KnownFormats, SwaggerSchema(true)).Validate(schema.Example))
	}
	if schema.Items != nil {
		if schema.Items.Schema != nil {
			res.Merge(ex.validateExampleValueSchemaAgainstSchema(path+".items.example", in, schema.Items.Schema))
		}
		// Multiple schemas in items
		if schema.Items.Schemas != nil { // Safeguard
			for i, sch := range schema.Items.Schemas {
				res.Merge(ex.validateExampleValueSchemaAgainstSchema(fmt.Sprintf("%s.items[%d].example", path, i), in, &sch))
			}
		}
	}
	if _, err := compileRegexp(schema.Pattern); err != nil {
		res.AddErrors(invalidPatternInMsg(path, in, schema.Pattern))
	}
	if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
		// NOTE: we keep validating values, even though additionalItems is unsupported in Swagger 2.0 (and 3.0 as well)
		res.Merge(ex.validateExampleValueSchemaAgainstSchema(fmt.Sprintf("%s.additionalItems", path), in, schema.AdditionalItems.Schema))
	}
	for propName, prop := range schema.Properties {
		res.Merge(ex.validateExampleValueSchemaAgainstSchema(path+"."+propName, in, &prop))
	}
	for propName, prop := range schema.PatternProperties {
		res.Merge(ex.validateExampleValueSchemaAgainstSchema(path+"."+propName, in, &prop))
	}
	if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
		res.Merge(ex.validateExampleValueSchemaAgainstSchema(fmt.Sprintf("%s.additionalProperties", path), in, schema.AdditionalProperties.Schema))
	}
	if schema.AllOf != nil {
		for i, aoSch := range schema.AllOf {
			res.Merge(ex.validateExampleValueSchemaAgainstSchema(fmt.Sprintf("%s.allOf[%d]", path, i), in, &aoSch))
		}
	}
	return res
}

// TODO: Temporary duplicated code. Need to refactor with examples
// nolint: dupl
func (ex *exampleValidator) validateExampleValueItemsAgainstSchema(path, in string, root interface{}, items *spec.Items) *Result {
	res := new(Result)
	s := ex.SpecValidator
	if items != nil {
		if items.Example != nil {
			res.MergeAsWarnings(newItemsValidator(path, in, items, root, s.KnownFormats).Validate(0, items.Example))
		}
		if items.Items != nil {
			res.Merge(ex.validateExampleValueItemsAgainstSchema(path+"[0].example", in, root, items.Items))
		}
		if _, err := compileRegexp(items.Pattern); err != nil {
			res.AddErrors(invalidPatternInMsg(path, in, items.Pattern))
		}
	}
	return res
}
