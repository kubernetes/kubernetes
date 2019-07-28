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

package analysis

import (
	"fmt"
	slashpath "path"
	"strconv"
	"strings"

	"github.com/go-openapi/jsonpointer"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/swag"
)

type referenceAnalysis struct {
	schemas        map[string]spec.Ref
	responses      map[string]spec.Ref
	parameters     map[string]spec.Ref
	items          map[string]spec.Ref
	headerItems    map[string]spec.Ref
	parameterItems map[string]spec.Ref
	allRefs        map[string]spec.Ref
	pathItems      map[string]spec.Ref
}

func (r *referenceAnalysis) addRef(key string, ref spec.Ref) {
	r.allRefs["#"+key] = ref
}

func (r *referenceAnalysis) addItemsRef(key string, items *spec.Items, location string) {
	r.items["#"+key] = items.Ref
	r.addRef(key, items.Ref)
	if location == "header" {
		// NOTE: in swagger 2.0, headers and parameters (but not body param schemas) are simple schemas
		// and $ref are not supported here. However it is possible to analyze this.
		r.headerItems["#"+key] = items.Ref
	} else {
		r.parameterItems["#"+key] = items.Ref
	}
}

func (r *referenceAnalysis) addSchemaRef(key string, ref SchemaRef) {
	r.schemas["#"+key] = ref.Schema.Ref
	r.addRef(key, ref.Schema.Ref)
}

func (r *referenceAnalysis) addResponseRef(key string, resp *spec.Response) {
	r.responses["#"+key] = resp.Ref
	r.addRef(key, resp.Ref)
}

func (r *referenceAnalysis) addParamRef(key string, param *spec.Parameter) {
	r.parameters["#"+key] = param.Ref
	r.addRef(key, param.Ref)
}

func (r *referenceAnalysis) addPathItemRef(key string, pathItem *spec.PathItem) {
	r.pathItems["#"+key] = pathItem.Ref
	r.addRef(key, pathItem.Ref)
}

type patternAnalysis struct {
	parameters  map[string]string
	headers     map[string]string
	items       map[string]string
	schemas     map[string]string
	allPatterns map[string]string
}

func (p *patternAnalysis) addPattern(key, pattern string) {
	p.allPatterns["#"+key] = pattern
}

func (p *patternAnalysis) addParameterPattern(key, pattern string) {
	p.parameters["#"+key] = pattern
	p.addPattern(key, pattern)
}

func (p *patternAnalysis) addHeaderPattern(key, pattern string) {
	p.headers["#"+key] = pattern
	p.addPattern(key, pattern)
}

func (p *patternAnalysis) addItemsPattern(key, pattern string) {
	p.items["#"+key] = pattern
	p.addPattern(key, pattern)
}

func (p *patternAnalysis) addSchemaPattern(key, pattern string) {
	p.schemas["#"+key] = pattern
	p.addPattern(key, pattern)
}

type enumAnalysis struct {
	parameters map[string][]interface{}
	headers    map[string][]interface{}
	items      map[string][]interface{}
	schemas    map[string][]interface{}
	allEnums   map[string][]interface{}
}

func (p *enumAnalysis) addEnum(key string, enum []interface{}) {
	p.allEnums["#"+key] = enum
}

func (p *enumAnalysis) addParameterEnum(key string, enum []interface{}) {
	p.parameters["#"+key] = enum
	p.addEnum(key, enum)
}

func (p *enumAnalysis) addHeaderEnum(key string, enum []interface{}) {
	p.headers["#"+key] = enum
	p.addEnum(key, enum)
}

func (p *enumAnalysis) addItemsEnum(key string, enum []interface{}) {
	p.items["#"+key] = enum
	p.addEnum(key, enum)
}

func (p *enumAnalysis) addSchemaEnum(key string, enum []interface{}) {
	p.schemas["#"+key] = enum
	p.addEnum(key, enum)
}

// New takes a swagger spec object and returns an analyzed spec document.
// The analyzed document contains a number of indices that make it easier to
// reason about semantics of a swagger specification for use in code generation
// or validation etc.
func New(doc *spec.Swagger) *Spec {
	a := &Spec{
		spec:        doc,
		consumes:    make(map[string]struct{}, 150),
		produces:    make(map[string]struct{}, 150),
		authSchemes: make(map[string]struct{}, 150),
		operations:  make(map[string]map[string]*spec.Operation, 150),
		allSchemas:  make(map[string]SchemaRef, 150),
		allOfs:      make(map[string]SchemaRef, 150),
		references: referenceAnalysis{
			schemas:        make(map[string]spec.Ref, 150),
			pathItems:      make(map[string]spec.Ref, 150),
			responses:      make(map[string]spec.Ref, 150),
			parameters:     make(map[string]spec.Ref, 150),
			items:          make(map[string]spec.Ref, 150),
			headerItems:    make(map[string]spec.Ref, 150),
			parameterItems: make(map[string]spec.Ref, 150),
			allRefs:        make(map[string]spec.Ref, 150),
		},
		patterns: patternAnalysis{
			parameters:  make(map[string]string, 150),
			headers:     make(map[string]string, 150),
			items:       make(map[string]string, 150),
			schemas:     make(map[string]string, 150),
			allPatterns: make(map[string]string, 150),
		},
		enums: enumAnalysis{
			parameters: make(map[string][]interface{}, 150),
			headers:    make(map[string][]interface{}, 150),
			items:      make(map[string][]interface{}, 150),
			schemas:    make(map[string][]interface{}, 150),
			allEnums:   make(map[string][]interface{}, 150),
		},
	}
	a.initialize()
	return a
}

// Spec is an analyzed specification object. It takes a swagger spec object and turns it into a registry
// with a bunch of utility methods to act on the information in the spec.
type Spec struct {
	spec        *spec.Swagger
	consumes    map[string]struct{}
	produces    map[string]struct{}
	authSchemes map[string]struct{}
	operations  map[string]map[string]*spec.Operation
	references  referenceAnalysis
	patterns    patternAnalysis
	enums       enumAnalysis
	allSchemas  map[string]SchemaRef
	allOfs      map[string]SchemaRef
}

func (s *Spec) reset() {
	s.consumes = make(map[string]struct{}, 150)
	s.produces = make(map[string]struct{}, 150)
	s.authSchemes = make(map[string]struct{}, 150)
	s.operations = make(map[string]map[string]*spec.Operation, 150)
	s.allSchemas = make(map[string]SchemaRef, 150)
	s.allOfs = make(map[string]SchemaRef, 150)
	s.references.schemas = make(map[string]spec.Ref, 150)
	s.references.pathItems = make(map[string]spec.Ref, 150)
	s.references.responses = make(map[string]spec.Ref, 150)
	s.references.parameters = make(map[string]spec.Ref, 150)
	s.references.items = make(map[string]spec.Ref, 150)
	s.references.headerItems = make(map[string]spec.Ref, 150)
	s.references.parameterItems = make(map[string]spec.Ref, 150)
	s.references.allRefs = make(map[string]spec.Ref, 150)
	s.patterns.parameters = make(map[string]string, 150)
	s.patterns.headers = make(map[string]string, 150)
	s.patterns.items = make(map[string]string, 150)
	s.patterns.schemas = make(map[string]string, 150)
	s.patterns.allPatterns = make(map[string]string, 150)
	s.enums.parameters = make(map[string][]interface{}, 150)
	s.enums.headers = make(map[string][]interface{}, 150)
	s.enums.items = make(map[string][]interface{}, 150)
	s.enums.schemas = make(map[string][]interface{}, 150)
	s.enums.allEnums = make(map[string][]interface{}, 150)
}

func (s *Spec) reload() {
	s.reset()
	s.initialize()
}

func (s *Spec) initialize() {
	for _, c := range s.spec.Consumes {
		s.consumes[c] = struct{}{}
	}
	for _, c := range s.spec.Produces {
		s.produces[c] = struct{}{}
	}
	for _, ss := range s.spec.Security {
		for k := range ss {
			s.authSchemes[k] = struct{}{}
		}
	}
	for path, pathItem := range s.AllPaths() {
		s.analyzeOperations(path, &pathItem)
	}

	for name, parameter := range s.spec.Parameters {
		refPref := slashpath.Join("/parameters", jsonpointer.Escape(name))
		if parameter.Items != nil {
			s.analyzeItems("items", parameter.Items, refPref, "parameter")
		}
		if parameter.In == "body" && parameter.Schema != nil {
			s.analyzeSchema("schema", *parameter.Schema, refPref)
		}
		if parameter.Pattern != "" {
			s.patterns.addParameterPattern(refPref, parameter.Pattern)
		}
		if len(parameter.Enum) > 0 {
			s.enums.addParameterEnum(refPref, parameter.Enum)
		}
	}

	for name, response := range s.spec.Responses {
		refPref := slashpath.Join("/responses", jsonpointer.Escape(name))
		for k, v := range response.Headers {
			hRefPref := slashpath.Join(refPref, "headers", k)
			if v.Items != nil {
				s.analyzeItems("items", v.Items, hRefPref, "header")
			}
			if v.Pattern != "" {
				s.patterns.addHeaderPattern(hRefPref, v.Pattern)
			}
			if len(v.Enum) > 0 {
				s.enums.addHeaderEnum(hRefPref, v.Enum)
			}
		}
		if response.Schema != nil {
			s.analyzeSchema("schema", *response.Schema, refPref)
		}
	}

	for name, schema := range s.spec.Definitions {
		s.analyzeSchema(name, schema, "/definitions")
	}
	// TODO: after analyzing all things and flattening schemas etc
	// resolve all the collected references to their final representations
	// best put in a separate method because this could get expensive
}

func (s *Spec) analyzeOperations(path string, pi *spec.PathItem) {
	// TODO: resolve refs here?
	// Currently, operations declared via pathItem $ref are known only after expansion
	op := pi
	if pi.Ref.String() != "" {
		key := slashpath.Join("/paths", jsonpointer.Escape(path))
		s.references.addPathItemRef(key, pi)
	}
	s.analyzeOperation("GET", path, op.Get)
	s.analyzeOperation("PUT", path, op.Put)
	s.analyzeOperation("POST", path, op.Post)
	s.analyzeOperation("PATCH", path, op.Patch)
	s.analyzeOperation("DELETE", path, op.Delete)
	s.analyzeOperation("HEAD", path, op.Head)
	s.analyzeOperation("OPTIONS", path, op.Options)
	for i, param := range op.Parameters {
		refPref := slashpath.Join("/paths", jsonpointer.Escape(path), "parameters", strconv.Itoa(i))
		if param.Ref.String() != "" {
			s.references.addParamRef(refPref, &param)
		}
		if param.Pattern != "" {
			s.patterns.addParameterPattern(refPref, param.Pattern)
		}
		if len(param.Enum) > 0 {
			s.enums.addParameterEnum(refPref, param.Enum)
		}
		if param.Items != nil {
			s.analyzeItems("items", param.Items, refPref, "parameter")
		}
		if param.Schema != nil {
			s.analyzeSchema("schema", *param.Schema, refPref)
		}
	}
}

func (s *Spec) analyzeItems(name string, items *spec.Items, prefix, location string) {
	if items == nil {
		return
	}
	refPref := slashpath.Join(prefix, name)
	s.analyzeItems(name, items.Items, refPref, location)
	if items.Ref.String() != "" {
		s.references.addItemsRef(refPref, items, location)
	}
	if items.Pattern != "" {
		s.patterns.addItemsPattern(refPref, items.Pattern)
	}
	if len(items.Enum) > 0 {
		s.enums.addItemsEnum(refPref, items.Enum)
	}
}

func (s *Spec) analyzeOperation(method, path string, op *spec.Operation) {
	if op == nil {
		return
	}

	for _, c := range op.Consumes {
		s.consumes[c] = struct{}{}
	}
	for _, c := range op.Produces {
		s.produces[c] = struct{}{}
	}
	for _, ss := range op.Security {
		for k := range ss {
			s.authSchemes[k] = struct{}{}
		}
	}
	if _, ok := s.operations[method]; !ok {
		s.operations[method] = make(map[string]*spec.Operation)
	}
	s.operations[method][path] = op
	prefix := slashpath.Join("/paths", jsonpointer.Escape(path), strings.ToLower(method))
	for i, param := range op.Parameters {
		refPref := slashpath.Join(prefix, "parameters", strconv.Itoa(i))
		if param.Ref.String() != "" {
			s.references.addParamRef(refPref, &param)
		}
		if param.Pattern != "" {
			s.patterns.addParameterPattern(refPref, param.Pattern)
		}
		if len(param.Enum) > 0 {
			s.enums.addParameterEnum(refPref, param.Enum)
		}
		s.analyzeItems("items", param.Items, refPref, "parameter")
		if param.In == "body" && param.Schema != nil {
			s.analyzeSchema("schema", *param.Schema, refPref)
		}
	}
	if op.Responses != nil {
		if op.Responses.Default != nil {
			refPref := slashpath.Join(prefix, "responses", "default")
			if op.Responses.Default.Ref.String() != "" {
				s.references.addResponseRef(refPref, op.Responses.Default)
			}
			for k, v := range op.Responses.Default.Headers {
				hRefPref := slashpath.Join(refPref, "headers", k)
				s.analyzeItems("items", v.Items, hRefPref, "header")
				if v.Pattern != "" {
					s.patterns.addHeaderPattern(hRefPref, v.Pattern)
				}
			}
			if op.Responses.Default.Schema != nil {
				s.analyzeSchema("schema", *op.Responses.Default.Schema, refPref)
			}
		}
		for k, res := range op.Responses.StatusCodeResponses {
			refPref := slashpath.Join(prefix, "responses", strconv.Itoa(k))
			if res.Ref.String() != "" {
				s.references.addResponseRef(refPref, &res)
			}
			for k, v := range res.Headers {
				hRefPref := slashpath.Join(refPref, "headers", k)
				s.analyzeItems("items", v.Items, hRefPref, "header")
				if v.Pattern != "" {
					s.patterns.addHeaderPattern(hRefPref, v.Pattern)
				}
				if len(v.Enum) > 0 {
					s.enums.addHeaderEnum(hRefPref, v.Enum)
				}
			}
			if res.Schema != nil {
				s.analyzeSchema("schema", *res.Schema, refPref)
			}
		}
	}
}

func (s *Spec) analyzeSchema(name string, schema spec.Schema, prefix string) {
	refURI := slashpath.Join(prefix, jsonpointer.Escape(name))
	schRef := SchemaRef{
		Name:     name,
		Schema:   &schema,
		Ref:      spec.MustCreateRef("#" + refURI),
		TopLevel: prefix == "/definitions",
	}

	s.allSchemas["#"+refURI] = schRef

	if schema.Ref.String() != "" {
		s.references.addSchemaRef(refURI, schRef)
	}
	if schema.Pattern != "" {
		s.patterns.addSchemaPattern(refURI, schema.Pattern)
	}
	if len(schema.Enum) > 0 {
		s.enums.addSchemaEnum(refURI, schema.Enum)
	}

	for k, v := range schema.Definitions {
		s.analyzeSchema(k, v, slashpath.Join(refURI, "definitions"))
	}
	for k, v := range schema.Properties {
		s.analyzeSchema(k, v, slashpath.Join(refURI, "properties"))
	}
	for k, v := range schema.PatternProperties {
		// NOTE: swagger 2.0 does not support PatternProperties.
		// However it is possible to analyze this in a schema
		s.analyzeSchema(k, v, slashpath.Join(refURI, "patternProperties"))
	}
	for i, v := range schema.AllOf {
		s.analyzeSchema(strconv.Itoa(i), v, slashpath.Join(refURI, "allOf"))
	}
	if len(schema.AllOf) > 0 {
		s.allOfs["#"+refURI] = schRef
	}
	for i, v := range schema.AnyOf {
		// NOTE: swagger 2.0 does not support anyOf constructs.
		// However it is possible to analyze this in a schema
		s.analyzeSchema(strconv.Itoa(i), v, slashpath.Join(refURI, "anyOf"))
	}
	for i, v := range schema.OneOf {
		// NOTE: swagger 2.0 does not support oneOf constructs.
		// However it is possible to analyze this in a schema
		s.analyzeSchema(strconv.Itoa(i), v, slashpath.Join(refURI, "oneOf"))
	}
	if schema.Not != nil {
		// NOTE: swagger 2.0 does not support "not" constructs.
		// However it is possible to analyze this in a schema
		s.analyzeSchema("not", *schema.Not, refURI)
	}
	if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
		s.analyzeSchema("additionalProperties", *schema.AdditionalProperties.Schema, refURI)
	}
	if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
		// NOTE: swagger 2.0 does not support AdditionalItems.
		// However it is possible to analyze this in a schema
		s.analyzeSchema("additionalItems", *schema.AdditionalItems.Schema, refURI)
	}
	if schema.Items != nil {
		if schema.Items.Schema != nil {
			s.analyzeSchema("items", *schema.Items.Schema, refURI)
		}
		for i, sch := range schema.Items.Schemas {
			s.analyzeSchema(strconv.Itoa(i), sch, slashpath.Join(refURI, "items"))
		}
	}
}

// SecurityRequirement is a representation of a security requirement for an operation
type SecurityRequirement struct {
	Name   string
	Scopes []string
}

// SecurityRequirementsFor gets the security requirements for the operation
func (s *Spec) SecurityRequirementsFor(operation *spec.Operation) [][]SecurityRequirement {
	if s.spec.Security == nil && operation.Security == nil {
		return nil
	}

	schemes := s.spec.Security
	if operation.Security != nil {
		schemes = operation.Security
	}

	result := [][]SecurityRequirement{}
	for _, scheme := range schemes {
		if len(scheme) == 0 {
			// append a zero object for anonymous
			result = append(result, []SecurityRequirement{{}})
			continue
		}
		var reqs []SecurityRequirement
		for k, v := range scheme {
			if v == nil {
				v = []string{}
			}
			reqs = append(reqs, SecurityRequirement{Name: k, Scopes: v})
		}
		result = append(result, reqs)
	}
	return result
}

// SecurityDefinitionsForRequirements gets the matching security definitions for a set of requirements
func (s *Spec) SecurityDefinitionsForRequirements(requirements []SecurityRequirement) map[string]spec.SecurityScheme {
	result := make(map[string]spec.SecurityScheme)

	for _, v := range requirements {
		if definition, ok := s.spec.SecurityDefinitions[v.Name]; ok {
			if definition != nil {
				result[v.Name] = *definition
			}
		}
	}
	return result
}

// SecurityDefinitionsFor gets the matching security definitions for a set of requirements
func (s *Spec) SecurityDefinitionsFor(operation *spec.Operation) map[string]spec.SecurityScheme {
	requirements := s.SecurityRequirementsFor(operation)
	if len(requirements) == 0 {
		return nil
	}

	result := make(map[string]spec.SecurityScheme)
	for _, reqs := range requirements {
		for _, v := range reqs {
			if v.Name == "" {
				// optional requirement
				continue
			}
			if _, ok := result[v.Name]; ok {
				// duplicate requirement
				continue
			}
			if definition, ok := s.spec.SecurityDefinitions[v.Name]; ok {
				if definition != nil {
					result[v.Name] = *definition
				}
			}
		}
	}
	return result
}

// ConsumesFor gets the mediatypes for the operation
func (s *Spec) ConsumesFor(operation *spec.Operation) []string {

	if len(operation.Consumes) == 0 {
		cons := make(map[string]struct{}, len(s.spec.Consumes))
		for _, k := range s.spec.Consumes {
			cons[k] = struct{}{}
		}
		return s.structMapKeys(cons)
	}

	cons := make(map[string]struct{}, len(operation.Consumes))
	for _, c := range operation.Consumes {
		cons[c] = struct{}{}
	}
	return s.structMapKeys(cons)
}

// ProducesFor gets the mediatypes for the operation
func (s *Spec) ProducesFor(operation *spec.Operation) []string {
	if len(operation.Produces) == 0 {
		prod := make(map[string]struct{}, len(s.spec.Produces))
		for _, k := range s.spec.Produces {
			prod[k] = struct{}{}
		}
		return s.structMapKeys(prod)
	}

	prod := make(map[string]struct{}, len(operation.Produces))
	for _, c := range operation.Produces {
		prod[c] = struct{}{}
	}
	return s.structMapKeys(prod)
}

func mapKeyFromParam(param *spec.Parameter) string {
	return fmt.Sprintf("%s#%s", param.In, fieldNameFromParam(param))
}

func fieldNameFromParam(param *spec.Parameter) string {
	// TODO: this should be x-go-name
	if nm, ok := param.Extensions.GetString("go-name"); ok {
		return nm
	}
	return swag.ToGoName(param.Name)
}

// ErrorOnParamFunc is a callback function to be invoked
// whenever an error is encountered while resolving references
// on parameters.
//
// This function takes as input the spec.Parameter which triggered the
// error and the error itself.
//
// If the callback function returns false, the calling function should bail.
//
// If it returns true, the calling function should continue evaluating parameters.
// A nil ErrorOnParamFunc must be evaluated as equivalent to panic().
type ErrorOnParamFunc func(spec.Parameter, error) bool

func (s *Spec) paramsAsMap(parameters []spec.Parameter, res map[string]spec.Parameter, callmeOnError ErrorOnParamFunc) {
	for _, param := range parameters {
		pr := param
		if pr.Ref.String() != "" {
			obj, _, err := pr.Ref.GetPointer().Get(s.spec)
			if err != nil {
				if callmeOnError != nil {
					if callmeOnError(param, fmt.Errorf("invalid reference: %q", pr.Ref.String())) {
						continue
					}
					break
				} else {
					panic(fmt.Sprintf("invalid reference: %q", pr.Ref.String()))
				}
			}
			if objAsParam, ok := obj.(spec.Parameter); ok {
				pr = objAsParam
			} else {
				if callmeOnError != nil {
					if callmeOnError(param, fmt.Errorf("resolved reference is not a parameter: %q", pr.Ref.String())) {
						continue
					}
					break
				} else {
					panic(fmt.Sprintf("resolved reference is not a parameter: %q", pr.Ref.String()))
				}
			}
		}
		res[mapKeyFromParam(&pr)] = pr
	}
}

// ParametersFor the specified operation id.
//
// Assumes parameters properly resolve references if any and that
// such references actually resolve to a parameter object.
// Otherwise, panics.
func (s *Spec) ParametersFor(operationID string) []spec.Parameter {
	return s.SafeParametersFor(operationID, nil)
}

// SafeParametersFor the specified operation id.
//
// Does not assume parameters properly resolve references or that
// such references actually resolve to a parameter object.
//
// Upon error, invoke a ErrorOnParamFunc callback with the erroneous
// parameters. If the callback is set to nil, panics upon errors.
func (s *Spec) SafeParametersFor(operationID string, callmeOnError ErrorOnParamFunc) []spec.Parameter {
	gatherParams := func(pi *spec.PathItem, op *spec.Operation) []spec.Parameter {
		bag := make(map[string]spec.Parameter)
		s.paramsAsMap(pi.Parameters, bag, callmeOnError)
		s.paramsAsMap(op.Parameters, bag, callmeOnError)

		var res []spec.Parameter
		for _, v := range bag {
			res = append(res, v)
		}
		return res
	}
	for _, pi := range s.spec.Paths.Paths {
		if pi.Get != nil && pi.Get.ID == operationID {
			return gatherParams(&pi, pi.Get)
		}
		if pi.Head != nil && pi.Head.ID == operationID {
			return gatherParams(&pi, pi.Head)
		}
		if pi.Options != nil && pi.Options.ID == operationID {
			return gatherParams(&pi, pi.Options)
		}
		if pi.Post != nil && pi.Post.ID == operationID {
			return gatherParams(&pi, pi.Post)
		}
		if pi.Patch != nil && pi.Patch.ID == operationID {
			return gatherParams(&pi, pi.Patch)
		}
		if pi.Put != nil && pi.Put.ID == operationID {
			return gatherParams(&pi, pi.Put)
		}
		if pi.Delete != nil && pi.Delete.ID == operationID {
			return gatherParams(&pi, pi.Delete)
		}
	}
	return nil
}

// ParamsFor the specified method and path. Aggregates them with the defaults etc, so it's all the params that
// apply for the method and path.
//
// Assumes parameters properly resolve references if any and that
// such references actually resolve to a parameter object.
// Otherwise, panics.
func (s *Spec) ParamsFor(method, path string) map[string]spec.Parameter {
	return s.SafeParamsFor(method, path, nil)
}

// SafeParamsFor the specified method and path. Aggregates them with the defaults etc, so it's all the params that
// apply for the method and path.
//
// Does not assume parameters properly resolve references or that
// such references actually resolve to a parameter object.
//
// Upon error, invoke a ErrorOnParamFunc callback with the erroneous
// parameters. If the callback is set to nil, panics upon errors.
func (s *Spec) SafeParamsFor(method, path string, callmeOnError ErrorOnParamFunc) map[string]spec.Parameter {
	res := make(map[string]spec.Parameter)
	if pi, ok := s.spec.Paths.Paths[path]; ok {
		s.paramsAsMap(pi.Parameters, res, callmeOnError)
		s.paramsAsMap(s.operations[strings.ToUpper(method)][path].Parameters, res, callmeOnError)
	}
	return res
}

// OperationForName gets the operation for the given id
func (s *Spec) OperationForName(operationID string) (string, string, *spec.Operation, bool) {
	for method, pathItem := range s.operations {
		for path, op := range pathItem {
			if operationID == op.ID {
				return method, path, op, true
			}
		}
	}
	return "", "", nil, false
}

// OperationFor the given method and path
func (s *Spec) OperationFor(method, path string) (*spec.Operation, bool) {
	if mp, ok := s.operations[strings.ToUpper(method)]; ok {
		op, fn := mp[path]
		return op, fn
	}
	return nil, false
}

// Operations gathers all the operations specified in the spec document
func (s *Spec) Operations() map[string]map[string]*spec.Operation {
	return s.operations
}

func (s *Spec) structMapKeys(mp map[string]struct{}) []string {
	if len(mp) == 0 {
		return nil
	}

	result := make([]string, 0, len(mp))
	for k := range mp {
		result = append(result, k)
	}
	return result
}

// AllPaths returns all the paths in the swagger spec
func (s *Spec) AllPaths() map[string]spec.PathItem {
	if s.spec == nil || s.spec.Paths == nil {
		return nil
	}
	return s.spec.Paths.Paths
}

// OperationIDs gets all the operation ids based on method an dpath
func (s *Spec) OperationIDs() []string {
	if len(s.operations) == 0 {
		return nil
	}
	result := make([]string, 0, len(s.operations))
	for method, v := range s.operations {
		for p, o := range v {
			if o.ID != "" {
				result = append(result, o.ID)
			} else {
				result = append(result, fmt.Sprintf("%s %s", strings.ToUpper(method), p))
			}
		}
	}
	return result
}

// OperationMethodPaths gets all the operation ids based on method an dpath
func (s *Spec) OperationMethodPaths() []string {
	if len(s.operations) == 0 {
		return nil
	}
	result := make([]string, 0, len(s.operations))
	for method, v := range s.operations {
		for p := range v {
			result = append(result, fmt.Sprintf("%s %s", strings.ToUpper(method), p))
		}
	}
	return result
}

// RequiredConsumes gets all the distinct consumes that are specified in the specification document
func (s *Spec) RequiredConsumes() []string {
	return s.structMapKeys(s.consumes)
}

// RequiredProduces gets all the distinct produces that are specified in the specification document
func (s *Spec) RequiredProduces() []string {
	return s.structMapKeys(s.produces)
}

// RequiredSecuritySchemes gets all the distinct security schemes that are specified in the swagger spec
func (s *Spec) RequiredSecuritySchemes() []string {
	return s.structMapKeys(s.authSchemes)
}

// SchemaRef is a reference to a schema
type SchemaRef struct {
	Name     string
	Ref      spec.Ref
	Schema   *spec.Schema
	TopLevel bool
}

// SchemasWithAllOf returns schema references to all schemas that are defined
// with an allOf key
func (s *Spec) SchemasWithAllOf() (result []SchemaRef) {
	for _, v := range s.allOfs {
		result = append(result, v)
	}
	return
}

// AllDefinitions returns schema references for all the definitions that were discovered
func (s *Spec) AllDefinitions() (result []SchemaRef) {
	for _, v := range s.allSchemas {
		result = append(result, v)
	}
	return
}

// AllDefinitionReferences returns json refs for all the discovered schemas
func (s *Spec) AllDefinitionReferences() (result []string) {
	for _, v := range s.references.schemas {
		result = append(result, v.String())
	}
	return
}

// AllParameterReferences returns json refs for all the discovered parameters
func (s *Spec) AllParameterReferences() (result []string) {
	for _, v := range s.references.parameters {
		result = append(result, v.String())
	}
	return
}

// AllResponseReferences returns json refs for all the discovered responses
func (s *Spec) AllResponseReferences() (result []string) {
	for _, v := range s.references.responses {
		result = append(result, v.String())
	}
	return
}

// AllPathItemReferences returns the references for all the items
func (s *Spec) AllPathItemReferences() (result []string) {
	for _, v := range s.references.pathItems {
		result = append(result, v.String())
	}
	return
}

// AllItemsReferences returns the references for all the items in simple schemas (parameters or headers).
//
// NOTE: since Swagger 2.0 forbids $ref in simple params, this should always yield an empty slice for a valid
// Swagger 2.0 spec.
func (s *Spec) AllItemsReferences() (result []string) {
	for _, v := range s.references.items {
		result = append(result, v.String())
	}
	return
}

// AllReferences returns all the references found in the document, with possible duplicates
func (s *Spec) AllReferences() (result []string) {
	for _, v := range s.references.allRefs {
		result = append(result, v.String())
	}
	return
}

// AllRefs returns all the unique references found in the document
func (s *Spec) AllRefs() (result []spec.Ref) {
	set := make(map[string]struct{})
	for _, v := range s.references.allRefs {
		a := v.String()
		if a == "" {
			continue
		}
		if _, ok := set[a]; !ok {
			set[a] = struct{}{}
			result = append(result, v)
		}
	}
	return
}

func cloneStringMap(source map[string]string) map[string]string {
	res := make(map[string]string, len(source))
	for k, v := range source {
		res[k] = v
	}
	return res
}

func cloneEnumMap(source map[string][]interface{}) map[string][]interface{} {
	res := make(map[string][]interface{}, len(source))
	for k, v := range source {
		res[k] = v
	}
	return res
}

// ParameterPatterns returns all the patterns found in parameters
// the map is cloned to avoid accidental changes
func (s *Spec) ParameterPatterns() map[string]string {
	return cloneStringMap(s.patterns.parameters)
}

// HeaderPatterns returns all the patterns found in response headers
// the map is cloned to avoid accidental changes
func (s *Spec) HeaderPatterns() map[string]string {
	return cloneStringMap(s.patterns.headers)
}

// ItemsPatterns returns all the patterns found in simple array items
// the map is cloned to avoid accidental changes
func (s *Spec) ItemsPatterns() map[string]string {
	return cloneStringMap(s.patterns.items)
}

// SchemaPatterns returns all the patterns found in schemas
// the map is cloned to avoid accidental changes
func (s *Spec) SchemaPatterns() map[string]string {
	return cloneStringMap(s.patterns.schemas)
}

// AllPatterns returns all the patterns found in the spec
// the map is cloned to avoid accidental changes
func (s *Spec) AllPatterns() map[string]string {
	return cloneStringMap(s.patterns.allPatterns)
}

// ParameterEnums returns all the enums found in parameters
// the map is cloned to avoid accidental changes
func (s *Spec) ParameterEnums() map[string][]interface{} {
	return cloneEnumMap(s.enums.parameters)
}

// HeaderEnums returns all the enums found in response headers
// the map is cloned to avoid accidental changes
func (s *Spec) HeaderEnums() map[string][]interface{} {
	return cloneEnumMap(s.enums.headers)
}

// ItemsEnums returns all the enums found in simple array items
// the map is cloned to avoid accidental changes
func (s *Spec) ItemsEnums() map[string][]interface{} {
	return cloneEnumMap(s.enums.items)
}

// SchemaEnums returns all the enums found in schemas
// the map is cloned to avoid accidental changes
func (s *Spec) SchemaEnums() map[string][]interface{} {
	return cloneEnumMap(s.enums.schemas)
}

// AllEnums returns all the enums found in the spec
// the map is cloned to avoid accidental changes
func (s *Spec) AllEnums() map[string][]interface{} {
	return cloneEnumMap(s.enums.allEnums)
}
