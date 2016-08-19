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
	schemas    map[string]spec.Ref
	responses  map[string]spec.Ref
	parameters map[string]spec.Ref
	items      map[string]spec.Ref
	allRefs    map[string]spec.Ref
	referenced struct {
		schemas    map[string]SchemaRef
		responses  map[string]*spec.Response
		parameters map[string]*spec.Parameter
	}
}

func (r *referenceAnalysis) addRef(key string, ref spec.Ref) {
	r.allRefs["#"+key] = ref
}

func (r *referenceAnalysis) addItemsRef(key string, items *spec.Items) {
	r.items["#"+key] = items.Ref
	r.addRef(key, items.Ref)
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
			schemas:    make(map[string]spec.Ref, 150),
			responses:  make(map[string]spec.Ref, 150),
			parameters: make(map[string]spec.Ref, 150),
			items:      make(map[string]spec.Ref, 150),
			allRefs:    make(map[string]spec.Ref, 150),
		},
	}
	a.references.referenced.schemas = make(map[string]SchemaRef, 150)
	a.references.referenced.responses = make(map[string]*spec.Response, 150)
	a.references.referenced.parameters = make(map[string]*spec.Parameter, 150)
	a.initialize()
	return a
}

// Spec takes a swagger spec object and turns it into a registry
// with a bunch of utility methods to act on the information in the spec
type Spec struct {
	spec        *spec.Swagger
	consumes    map[string]struct{}
	produces    map[string]struct{}
	authSchemes map[string]struct{}
	operations  map[string]map[string]*spec.Operation
	references  referenceAnalysis
	allSchemas  map[string]SchemaRef
	allOfs      map[string]SchemaRef
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
			s.analyzeItems("items", parameter.Items, refPref)
		}
		if parameter.In == "body" && parameter.Schema != nil {
			s.analyzeSchema("schema", *parameter.Schema, refPref)
		}
	}

	for name, response := range s.spec.Responses {
		refPref := slashpath.Join("/responses", jsonpointer.Escape(name))
		for _, v := range response.Headers {
			if v.Items != nil {
				s.analyzeItems("items", v.Items, refPref)
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
	op := pi
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
		if param.Items != nil {
			s.analyzeItems("items", param.Items, refPref)
		}
		if param.Schema != nil {
			s.analyzeSchema("schema", *param.Schema, refPref)
		}
	}
}

func (s *Spec) analyzeItems(name string, items *spec.Items, prefix string) {
	if items == nil {
		return
	}
	refPref := slashpath.Join(prefix, name)
	s.analyzeItems(name, items.Items, refPref)
	if items.Ref.String() != "" {
		s.references.addItemsRef(refPref, items)
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
		s.analyzeItems("items", param.Items, refPref)
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
			for _, v := range op.Responses.Default.Headers {
				s.analyzeItems("items", v.Items, refPref)
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
			for _, v := range res.Headers {
				s.analyzeItems("items", v.Items, refPref)
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
		Name:   name,
		Schema: &schema,
		Ref:    spec.MustCreateRef("#" + refURI),
	}
	s.allSchemas["#"+refURI] = schRef
	if schema.Ref.String() != "" {
		s.references.addSchemaRef(refURI, schRef)
	}
	for k, v := range schema.Definitions {
		s.analyzeSchema(k, v, slashpath.Join(refURI, "definitions"))
	}
	for k, v := range schema.Properties {
		s.analyzeSchema(k, v, slashpath.Join(refURI, "properties"))
	}
	for k, v := range schema.PatternProperties {
		s.analyzeSchema(k, v, slashpath.Join(refURI, "patternProperties"))
	}
	for i, v := range schema.AllOf {
		s.analyzeSchema(strconv.Itoa(i), v, slashpath.Join(refURI, "allOf"))
	}
	if len(schema.AllOf) > 0 {
		s.allOfs["#"+refURI] = SchemaRef{Name: name, Schema: &schema, Ref: spec.MustCreateRef("#" + refURI)}
	}
	for i, v := range schema.AnyOf {
		s.analyzeSchema(strconv.Itoa(i), v, slashpath.Join(refURI, "anyOf"))
	}
	for i, v := range schema.OneOf {
		s.analyzeSchema(strconv.Itoa(i), v, slashpath.Join(refURI, "oneOf"))
	}
	if schema.Not != nil {
		s.analyzeSchema("not", *schema.Not, refURI)
	}
	if schema.AdditionalProperties != nil && schema.AdditionalProperties.Schema != nil {
		s.analyzeSchema("additionalProperties", *schema.AdditionalProperties.Schema, refURI)
	}
	if schema.AdditionalItems != nil && schema.AdditionalItems.Schema != nil {
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
func (s *Spec) SecurityRequirementsFor(operation *spec.Operation) []SecurityRequirement {
	if s.spec.Security == nil && operation.Security == nil {
		return nil
	}

	schemes := s.spec.Security
	if operation.Security != nil {
		schemes = operation.Security
	}

	unique := make(map[string]SecurityRequirement)
	for _, scheme := range schemes {
		for k, v := range scheme {
			if _, ok := unique[k]; !ok {
				unique[k] = SecurityRequirement{Name: k, Scopes: v}
			}
		}
	}

	var result []SecurityRequirement
	for _, v := range unique {
		result = append(result, v)
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
	for _, v := range requirements {
		if definition, ok := s.spec.SecurityDefinitions[v.Name]; ok {
			if definition != nil {
				result[v.Name] = *definition
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
	if nm, ok := param.Extensions.GetString("go-name"); ok {
		return nm
	}
	return swag.ToGoName(param.Name)
}

func (s *Spec) paramsAsMap(parameters []spec.Parameter, res map[string]spec.Parameter) {
	for _, param := range parameters {
		pr := param
		if pr.Ref.String() != "" {
			obj, _, err := pr.Ref.GetPointer().Get(s.spec)
			if err != nil {
				panic(err)
			}
			pr = obj.(spec.Parameter)
		}
		res[mapKeyFromParam(&pr)] = pr
	}
}

// ParametersFor the specified operation id
func (s *Spec) ParametersFor(operationID string) []spec.Parameter {
	gatherParams := func(pi *spec.PathItem, op *spec.Operation) []spec.Parameter {
		bag := make(map[string]spec.Parameter)
		s.paramsAsMap(pi.Parameters, bag)
		s.paramsAsMap(op.Parameters, bag)

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
func (s *Spec) ParamsFor(method, path string) map[string]spec.Parameter {
	res := make(map[string]spec.Parameter)
	if pi, ok := s.spec.Paths.Paths[path]; ok {
		s.paramsAsMap(pi.Parameters, res)
		s.paramsAsMap(s.operations[strings.ToUpper(method)][path].Parameters, res)
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
	Name   string
	Ref    spec.Ref
	Schema *spec.Schema
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

// AllItemsReferences returns the references for all the items
func (s *Spec) AllItemsReferences() (result []string) {
	for _, v := range s.references.items {
		result = append(result, v.String())
	}
	return
}

// AllReferences returns all the references found in the document
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
