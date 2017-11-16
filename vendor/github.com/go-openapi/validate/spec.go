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
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"

	"github.com/go-openapi/analysis"
	"github.com/go-openapi/errors"
	"github.com/go-openapi/jsonpointer"
	"github.com/go-openapi/loads"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
)

// Spec validates a spec document
// It validates the spec json against the json schema for swagger
// and then validates a number of extra rules that can't be expressed in json schema:
//
// 	- definition can't declare a property that's already defined by one of its ancestors
// 	- definition's ancestor can't be a descendant of the same model
// 	- each api path should be non-verbatim (account for path param names) unique per method
// 	- each security reference should contain only unique scopes
// 	- each security scope in a security definition should be unique
// 	- each path parameter should correspond to a parameter placeholder and vice versa
// 	- each referencable definition must have references
// 	- each definition property listed in the required array must be defined in the properties of the model
// 	- each parameter should have a unique `name` and `type` combination
// 	- each operation should have only 1 parameter of type body
// 	- each reference must point to a valid object
// 	- every default value that is specified must validate against the schema for that property
// 	- items property is required for all schemas/definitions of type `array`
func Spec(doc *loads.Document, formats strfmt.Registry) error {
	errs, _ /*warns*/ := NewSpecValidator(doc.Schema(), formats).Validate(doc)
	if errs.HasErrors() {
		return errors.CompositeValidationError(errs.Errors...)
	}
	return nil
}

// AgainstSchema validates the specified data with the provided schema, when no schema
// is provided it uses the json schema as default
func AgainstSchema(schema *spec.Schema, data interface{}, formats strfmt.Registry) error {
	res := NewSchemaValidator(schema, nil, "", formats).Validate(data)
	if res.HasErrors() {
		return errors.CompositeValidationError(res.Errors...)
	}
	return nil
}

// SpecValidator validates a swagger spec
type SpecValidator struct {
	schema       *spec.Schema // swagger 2.0 schema
	spec         *loads.Document
	analyzer     *analysis.Spec
	expanded     *loads.Document
	KnownFormats strfmt.Registry
}

// NewSpecValidator creates a new swagger spec validator instance
func NewSpecValidator(schema *spec.Schema, formats strfmt.Registry) *SpecValidator {
	return &SpecValidator{
		schema:       schema,
		KnownFormats: formats,
	}
}

// Validate validates the swagger spec
func (s *SpecValidator) Validate(data interface{}) (errs *Result, warnings *Result) {
	var sd *loads.Document

	switch v := data.(type) {
	case *loads.Document:
		sd = v
	}
	if sd == nil {
		errs = sErr(errors.New(500, "spec validator can only validate spec.Document objects"))
		return
	}
	s.spec = sd
	s.analyzer = analysis.New(sd.Spec())

	errs = new(Result)
	warnings = new(Result)

	schv := NewSchemaValidator(s.schema, nil, "", s.KnownFormats)
	var obj interface{}
	if err := json.Unmarshal(sd.Raw(), &obj); err != nil {
		errs.AddErrors(err)
		return
	}
	errs.Merge(schv.Validate(obj)) // error -
	if errs.HasErrors() {
		return // no point in continuing
	}

	errs.Merge(s.validateReferencesValid()) // error -
	if errs.HasErrors() {
		return // no point in continuing
	}

	errs.Merge(s.validateDuplicateOperationIDs())
	errs.Merge(s.validateDuplicatePropertyNames())         // error -
	errs.Merge(s.validateParameters())                     // error -
	errs.Merge(s.validateItems())                          // error -
	errs.Merge(s.validateRequiredDefinitions())            // error -
	errs.Merge(s.validateDefaultValueValidAgainstSchema()) // error -
	errs.Merge(s.validateExamplesValidAgainstSchema())     // error -
	errs.Merge(s.validateNonEmptyPathParamNames())

	warnings.Merge(s.validateUniqueSecurityScopes()) // warning
	warnings.Merge(s.validateReferenced())           // warning

	return
}

func (s *SpecValidator) validateNonEmptyPathParamNames() *Result {
	res := new(Result)
	for k := range s.spec.Spec().Paths.Paths {
		if strings.Contains(k, "{}") {
			res.AddErrors(errors.New(422, "%q contains an empty path parameter", k))
		}
	}
	return res
}

func (s *SpecValidator) validateDuplicateOperationIDs() *Result {
	res := new(Result)
	known := make(map[string]int)
	for _, v := range s.analyzer.OperationIDs() {
		if v != "" {
			known[v]++
		}
	}
	for k, v := range known {
		if v > 1 {
			res.AddErrors(errors.New(422, "%q is defined %d times", k, v))
		}
	}
	return res
}

type dupProp struct {
	Name       string
	Definition string
}

func (s *SpecValidator) validateDuplicatePropertyNames() *Result {
	// definition can't declare a property that's already defined by one of its ancestors
	res := new(Result)
	for k, sch := range s.spec.Spec().Definitions {
		if len(sch.AllOf) == 0 {
			continue
		}

		knownanc := map[string]struct{}{
			"#/definitions/" + k: struct{}{},
		}

		ancs := s.validateCircularAncestry(k, sch, knownanc)
		if len(ancs) > 0 {
			res.AddErrors(errors.New(422, "definition %q has circular ancestry: %v", k, ancs))
			return res
		}

		knowns := make(map[string]struct{})
		dups := s.validateSchemaPropertyNames(k, sch, knowns)
		if len(dups) > 0 {
			var pns []string
			for _, v := range dups {
				pns = append(pns, v.Definition+"."+v.Name)
			}
			res.AddErrors(errors.New(422, "definition %q contains duplicate properties: %v", k, pns))
		}

	}
	return res
}

func (s *SpecValidator) validateSchemaPropertyNames(nm string, sch spec.Schema, knowns map[string]struct{}) []dupProp {
	var dups []dupProp

	schn := nm
	schc := &sch
	for schc.Ref.String() != "" {
		// gather property names
		reso, err := spec.ResolveRef(s.spec.Spec(), &schc.Ref)
		if err != nil {
			panic(err)
		}
		schc = reso
		schn = sch.Ref.String()
	}

	if len(schc.AllOf) > 0 {
		for _, chld := range schc.AllOf {
			dups = append(dups, s.validateSchemaPropertyNames(schn, chld, knowns)...)
		}
		return dups
	}

	for k := range schc.Properties {
		_, ok := knowns[k]
		if ok {
			dups = append(dups, dupProp{Name: k, Definition: schn})
		} else {
			knowns[k] = struct{}{}
		}
	}

	return dups
}

func (s *SpecValidator) validateCircularAncestry(nm string, sch spec.Schema, knowns map[string]struct{}) []string {
	if sch.Ref.String() == "" && len(sch.AllOf) == 0 {
		return nil
	}
	var ancs []string

	schn := nm
	schc := &sch
	for schc.Ref.String() != "" {
		reso, err := spec.ResolveRef(s.spec.Spec(), &schc.Ref)
		if err != nil {
			panic(err)
		}
		schc = reso
		schn = sch.Ref.String()
	}

	if schn != nm && schn != "" {
		if _, ok := knowns[schn]; ok {
			ancs = append(ancs, schn)
		}
		knowns[schn] = struct{}{}

		if len(ancs) > 0 {
			return ancs
		}
	}

	if len(schc.AllOf) > 0 {
		for _, chld := range schc.AllOf {
			if chld.Ref.String() != "" || len(chld.AllOf) > 0 {
				ancs = append(ancs, s.validateCircularAncestry(schn, chld, knowns)...)
				if len(ancs) > 0 {
					return ancs
				}
			}
		}
	}

	return ancs
}

func (s *SpecValidator) validateItems() *Result {
	// validate parameter, items, schema and response objects for presence of item if type is array
	res := new(Result)

	// TODO: implement support for lookups of refs
	for method, pi := range s.analyzer.Operations() {
		for path, op := range pi {
			for _, param := range s.analyzer.ParamsFor(method, path) {
				if param.TypeName() == "array" && param.ItemsTypeName() == "" {
					res.AddErrors(errors.New(422, "param %q for %q is a collection without an element type", param.Name, op.ID))
					continue
				}
				if param.In != "body" {
					if param.Items != nil {
						items := param.Items
						for items.TypeName() == "array" {
							if items.ItemsTypeName() == "" {
								res.AddErrors(errors.New(422, "param %q for %q is a collection without an element type", param.Name, op.ID))
								break
							}
							items = items.Items
						}
					}
				} else {
					if err := s.validateSchemaItems(*param.Schema, fmt.Sprintf("body param %q", param.Name), op.ID); err != nil {
						res.AddErrors(err)
					}
				}
			}

			var responses []spec.Response
			if op.Responses != nil {
				if op.Responses.Default != nil {
					responses = append(responses, *op.Responses.Default)
				}
				for _, v := range op.Responses.StatusCodeResponses {
					responses = append(responses, v)
				}
			}

			for _, resp := range responses {
				for hn, hv := range resp.Headers {
					if hv.TypeName() == "array" && hv.ItemsTypeName() == "" {
						res.AddErrors(errors.New(422, "header %q for %q is a collection without an element type", hn, op.ID))
					}
				}
				if resp.Schema != nil {
					if err := s.validateSchemaItems(*resp.Schema, "response body", op.ID); err != nil {
						res.AddErrors(err)
					}
				}
			}
		}
	}
	return res
}

func (s *SpecValidator) validateSchemaItems(schema spec.Schema, prefix, opID string) error {
	if !schema.Type.Contains("array") {
		return nil
	}

	if schema.Items == nil || schema.Items.Len() == 0 {
		return errors.New(422, "%s for %q is a collection without an element type", prefix, opID)
	}

	schemas := schema.Items.Schemas
	if schema.Items.Schema != nil {
		schemas = []spec.Schema{*schema.Items.Schema}
	}
	for _, sch := range schemas {
		if err := s.validateSchemaItems(sch, prefix, opID); err != nil {
			return err
		}
	}
	return nil
}

func (s *SpecValidator) validateUniqueSecurityScopes() *Result {
	// Each authorization/security reference should contain only unique scopes.
	// (Example: For an oauth2 authorization/security requirement, when listing the required scopes,
	// each scope should only be listed once.)

	return nil
}

func (s *SpecValidator) validatePathParamPresence(path string, fromPath, fromOperation []string) *Result {
	// Each defined operation path parameters must correspond to a named element in the API's path pattern.
	// (For example, you cannot have a path parameter named id for the following path /pets/{petId} but you must have a path parameter named petId.)
	res := new(Result)
	for _, l := range fromPath {
		var matched bool
		for _, r := range fromOperation {
			if l == "{"+r+"}" {
				matched = true
				break
			}
		}
		if !matched {
			res.Errors = append(res.Errors, errors.New(422, "path param %q has no parameter definition", l))
		}
	}

	for _, p := range fromOperation {
		var matched bool
		for _, r := range fromPath {
			if "{"+p+"}" == r {
				matched = true
				break
			}
		}
		if !matched {
			res.AddErrors(errors.New(422, "path param %q is not present in path %q", p, path))
		}
	}

	return res
}

func (s *SpecValidator) validateReferenced() *Result {
	var res Result
	res.Merge(s.validateReferencedParameters())
	res.Merge(s.validateReferencedResponses())
	res.Merge(s.validateReferencedDefinitions())
	return &res
}

func (s *SpecValidator) validateReferencedParameters() *Result {
	// Each referenceable definition must have references.
	params := s.spec.Spec().Parameters
	if len(params) == 0 {
		return nil
	}

	expected := make(map[string]struct{})
	for k := range params {
		expected["#/parameters/"+jsonpointer.Escape(k)] = struct{}{}
	}
	for _, k := range s.analyzer.AllParameterReferences() {
		if _, ok := expected[k]; ok {
			delete(expected, k)
		}
	}

	if len(expected) == 0 {
		return nil
	}
	var result Result
	for k := range expected {
		result.AddErrors(errors.New(422, "parameter %q is not used anywhere", k))
	}
	return &result
}

func (s *SpecValidator) validateReferencedResponses() *Result {
	// Each referenceable definition must have references.
	responses := s.spec.Spec().Responses
	if len(responses) == 0 {
		return nil
	}

	expected := make(map[string]struct{})
	for k := range responses {
		expected["#/responses/"+jsonpointer.Escape(k)] = struct{}{}
	}
	for _, k := range s.analyzer.AllResponseReferences() {
		if _, ok := expected[k]; ok {
			delete(expected, k)
		}
	}

	if len(expected) == 0 {
		return nil
	}
	var result Result
	for k := range expected {
		result.AddErrors(errors.New(422, "response %q is not used anywhere", k))
	}
	return &result
}

func (s *SpecValidator) validateReferencedDefinitions() *Result {
	// Each referenceable definition must have references.
	defs := s.spec.Spec().Definitions
	if len(defs) == 0 {
		return nil
	}

	expected := make(map[string]struct{})
	for k := range defs {
		expected["#/definitions/"+jsonpointer.Escape(k)] = struct{}{}
	}
	for _, k := range s.analyzer.AllDefinitionReferences() {
		if _, ok := expected[k]; ok {
			delete(expected, k)
		}
	}

	if len(expected) == 0 {
		return nil
	}
	var result Result
	for k := range expected {
		result.AddErrors(errors.New(422, "definition %q is not used anywhere", k))
	}
	return &result
}

func (s *SpecValidator) validateRequiredDefinitions() *Result {
	// Each definition property listed in the required array must be defined in the properties of the model
	res := new(Result)
	for d, v := range s.spec.Spec().Definitions {
	REQUIRED:
		for _, pn := range v.Required {
			if _, ok := v.Properties[pn]; ok {
				continue
			}

			for pp := range v.PatternProperties {
				re := regexp.MustCompile(pp)
				if re.MatchString(pn) {
					continue REQUIRED
				}
			}

			if v.AdditionalProperties != nil {
				if v.AdditionalProperties.Allows {
					continue
				}
				if v.AdditionalProperties.Schema != nil {
					continue
				}
			}

			res.AddErrors(errors.New(422, "%q is present in required but not defined as property in definition %q", pn, d))
		}
	}
	return res
}

func (s *SpecValidator) validateParameters() *Result {
	// each parameter should have a unique `name` and `type` combination
	// each operation should have only 1 parameter of type body
	// each api path should be non-verbatim (account for path param names) unique per method
	res := new(Result)
	for method, pi := range s.analyzer.Operations() {
		knownPaths := make(map[string]string)
		for path, op := range pi {
			segments, params := parsePath(path)
			knowns := make([]string, 0, len(segments))
			for _, s := range segments {
				knowns = append(knowns, s)
			}
			var fromPath []string
			for _, i := range params {
				fromPath = append(fromPath, knowns[i])
				knowns[i] = "!"
			}
			knownPath := strings.Join(knowns, "/")
			if orig, ok := knownPaths[knownPath]; ok {
				res.AddErrors(errors.New(422, "path %s overlaps with %s", path, orig))
			} else {
				knownPaths[knownPath] = path
			}

			ptypes := make(map[string]map[string]struct{})
			var firstBodyParam string
			sw := s.spec.Spec()
			var paramNames []string
		PARAMETERS:
			for _, ppr := range op.Parameters {
				pr := ppr
				for pr.Ref.String() != "" {
					obj, _, err := pr.Ref.GetPointer().Get(sw)
					if err != nil {
						log.Println(err)
						res.AddErrors(err)
						break PARAMETERS
					}
					pr = obj.(spec.Parameter)
				}
				pnames, ok := ptypes[pr.In]
				if !ok {
					pnames = make(map[string]struct{})
					ptypes[pr.In] = pnames
				}

				_, ok = pnames[pr.Name]
				if ok {
					res.AddErrors(errors.New(422, "duplicate parameter name %q for %q in operation %q", pr.Name, pr.In, op.ID))
				}
				pnames[pr.Name] = struct{}{}
			}

		PARAMETERS2:
			for _, ppr := range s.analyzer.ParamsFor(method, path) {
				pr := ppr
				for pr.Ref.String() != "" {
					obj, _, err := pr.Ref.GetPointer().Get(sw)
					if err != nil {
						res.AddErrors(err)
						break PARAMETERS2
					}
					pr = obj.(spec.Parameter)
				}

				if pr.In == "body" {
					if firstBodyParam != "" {
						res.AddErrors(errors.New(422, "operation %q has more than 1 body param (accepted: %q, dropped: %q)", op.ID, firstBodyParam, pr.Name))
					}
					firstBodyParam = pr.Name
				}

				if pr.In == "path" {
					paramNames = append(paramNames, pr.Name)
				}
			}
			res.Merge(s.validatePathParamPresence(path, fromPath, paramNames))
		}
	}
	return res
}

func parsePath(path string) (segments []string, params []int) {
	for i, p := range strings.Split(path, "/") {
		segments = append(segments, p)
		if len(p) > 0 && p[0] == '{' && p[len(p)-1] == '}' {
			params = append(params, i)
		}
	}
	return
}

func (s *SpecValidator) validateReferencesValid() *Result {
	// each reference must point to a valid object
	res := new(Result)
	for _, r := range s.analyzer.AllRefs() {
		if !r.IsValidURI() {
			res.AddErrors(errors.New(404, "invalid ref %q", r.String()))
		}
	}
	if !res.HasErrors() {
		exp, err := s.spec.Expanded()
		if err != nil {
			res.AddErrors(err)
		}
		s.expanded = exp
	}
	return res
}

func (s *SpecValidator) validateResponseExample(path string, r *spec.Response) *Result {
	res := new(Result)
	if r.Ref.String() != "" {
		nr, _, err := r.Ref.GetPointer().Get(s.spec.Spec())
		if err != nil {
			res.AddErrors(err)
			return res
		}
		rr := nr.(spec.Response)
		return s.validateResponseExample(path, &rr)
	}

	if r.Examples != nil {
		if r.Schema != nil {
			if example, ok := r.Examples["application/json"]; ok {
				res.Merge(NewSchemaValidator(r.Schema, s.spec.Spec(), path, s.KnownFormats).Validate(example))
			}

			// TODO: validate other media types too
		}
	}
	return res
}

func (s *SpecValidator) validateExamplesValidAgainstSchema() *Result {
	res := new(Result)

	for _, pathItem := range s.analyzer.Operations() {
		for path, op := range pathItem {
			if op.Responses.Default != nil {
				dr := op.Responses.Default
				res.Merge(s.validateResponseExample(path, dr))
			}
			for _, r := range op.Responses.StatusCodeResponses {
				res.Merge(s.validateResponseExample(path, &r))
			}
		}
	}

	return res
}

func (s *SpecValidator) validateDefaultValueValidAgainstSchema() *Result {
	// every default value that is specified must validate against the schema for that property
	// headers, items, parameters, schema

	res := new(Result)

	for method, pathItem := range s.analyzer.Operations() {
		for path, op := range pathItem {
			// parameters
			var hasForm, hasBody bool
		PARAMETERS:
			for _, pr := range s.analyzer.ParamsFor(method, path) {
				// expand ref is necessary
				param := pr
				for param.Ref.String() != "" {
					obj, _, err := param.Ref.GetPointer().Get(s.spec.Spec())
					if err != nil {
						res.AddErrors(err)
						break PARAMETERS
					}
					param = obj.(spec.Parameter)
				}
				if param.In == "formData" {
					if hasBody && !hasForm {
						res.AddErrors(errors.New(422, "operation %q has both formData and body parameters", op.ID))
					}
					hasForm = true
				}
				if param.In == "body" {
					if hasForm && !hasBody {
						res.AddErrors(errors.New(422, "operation %q has both body and formData parameters", op.ID))
					}
					hasBody = true
				}
				// check simple paramters first
				if param.Default != nil && param.Schema == nil {
					//fmt.Println(param.Name, "in", param.In, "has a default without a schema")
					// check param valid
					res.Merge(NewParamValidator(&param, s.KnownFormats).Validate(param.Default))
				}

				if param.Items != nil {
					res.Merge(s.validateDefaultValueItemsAgainstSchema(param.Name, param.In, &param, param.Items))
				}

				if param.Schema != nil {
					res.Merge(s.validateDefaultValueSchemaAgainstSchema(param.Name, param.In, param.Schema))
				}
			}

			if op.Responses.Default != nil {
				dr := op.Responses.Default
				for nm, h := range dr.Headers {
					if h.Default != nil {
						res.Merge(NewHeaderValidator(nm, &h, s.KnownFormats).Validate(h.Default))
					}
					if h.Items != nil {
						res.Merge(s.validateDefaultValueItemsAgainstSchema(nm, "header", &h, h.Items))
					}
				}
			}
			for _, r := range op.Responses.StatusCodeResponses {
				for nm, h := range r.Headers {
					if h.Default != nil {
						res.Merge(NewHeaderValidator(nm, &h, s.KnownFormats).Validate(h.Default))
					}
					if h.Items != nil {
						res.Merge(s.validateDefaultValueItemsAgainstSchema(nm, "header", &h, h.Items))
					}
				}
			}

		}
	}

	for nm, sch := range s.spec.Spec().Definitions {
		res.Merge(s.validateDefaultValueSchemaAgainstSchema(fmt.Sprintf("definitions.%s", nm), "body", &sch))
	}

	return res
}

func (s *SpecValidator) validateDefaultValueSchemaAgainstSchema(path, in string, schema *spec.Schema) *Result {
	res := new(Result)
	if schema != nil {
		if schema.Default != nil {
			res.Merge(NewSchemaValidator(schema, s.spec.Spec(), path, s.KnownFormats).Validate(schema.Default))
		}
		if schema.Items != nil {
			if schema.Items.Schema != nil {
				res.Merge(s.validateDefaultValueSchemaAgainstSchema(path+".items", in, schema.Items.Schema))
			}
			for i, sch := range schema.Items.Schemas {
				res.Merge(s.validateDefaultValueSchemaAgainstSchema(fmt.Sprintf("%s.items[%d]", path, i), in, &sch))
			}
		}
		if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
			res.Merge(s.validateDefaultValueSchemaAgainstSchema(fmt.Sprintf("%s.additionalItems", path), in, schema.AdditionalItems.Schema))
		}
		for propName, prop := range schema.Properties {
			res.Merge(s.validateDefaultValueSchemaAgainstSchema(path+"."+propName, in, &prop))
		}
		for propName, prop := range schema.PatternProperties {
			res.Merge(s.validateDefaultValueSchemaAgainstSchema(path+"."+propName, in, &prop))
		}
		if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
			res.Merge(s.validateDefaultValueSchemaAgainstSchema(fmt.Sprintf("%s.additionalProperties", path), in, schema.AdditionalProperties.Schema))
		}
		for i, aoSch := range schema.AllOf {
			res.Merge(s.validateDefaultValueSchemaAgainstSchema(fmt.Sprintf("%s.allOf[%d]", path, i), in, &aoSch))
		}

	}
	return res
}

func (s *SpecValidator) validateDefaultValueItemsAgainstSchema(path, in string, root interface{}, items *spec.Items) *Result {
	res := new(Result)
	if items != nil {
		if items.Default != nil {
			res.Merge(newItemsValidator(path, in, items, root, s.KnownFormats).Validate(0, items.Default))
		}
		if items.Items != nil {
			res.Merge(s.validateDefaultValueItemsAgainstSchema(path+"[0]", in, root, items.Items))
		}
	}
	return res
}
