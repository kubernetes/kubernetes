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
	"encoding/json"
	"fmt"
	"path/filepath"
	"sort"
	"testing"

	"github.com/go-openapi/loads/fmts"
	"github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
)

func schemeNames(schemes []SecurityRequirement) []string {
	var names []string
	for _, v := range schemes {
		names = append(names, v.Name)
	}
	sort.Sort(sort.StringSlice(names))
	return names
}

func TestAnalyzer(t *testing.T) {
	formatParam := spec.QueryParam("format").Typed("string", "")

	limitParam := spec.QueryParam("limit").Typed("integer", "int32")
	limitParam.Extensions = spec.Extensions(map[string]interface{}{})
	limitParam.Extensions.Add("go-name", "Limit")

	skipParam := spec.QueryParam("skip").Typed("integer", "int32")
	pi := spec.PathItem{}
	pi.Parameters = []spec.Parameter{*limitParam}

	op := &spec.Operation{}
	op.Consumes = []string{"application/x-yaml"}
	op.Produces = []string{"application/x-yaml"}
	op.Security = []map[string][]string{
		map[string][]string{"oauth2": []string{}},
		map[string][]string{"basic": nil},
	}
	op.ID = "someOperation"
	op.Parameters = []spec.Parameter{*skipParam}
	pi.Get = op

	pi2 := spec.PathItem{}
	pi2.Parameters = []spec.Parameter{*limitParam}
	op2 := &spec.Operation{}
	op2.ID = "anotherOperation"
	op2.Parameters = []spec.Parameter{*skipParam}
	pi2.Get = op2

	spec := &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Consumes: []string{"application/json"},
			Produces: []string{"application/json"},
			Security: []map[string][]string{
				map[string][]string{"apikey": nil},
			},
			SecurityDefinitions: map[string]*spec.SecurityScheme{
				"basic":  spec.BasicAuth(),
				"apiKey": spec.APIKeyAuth("api_key", "query"),
				"oauth2": spec.OAuth2AccessToken("http://authorize.com", "http://token.com"),
			},
			Parameters: map[string]spec.Parameter{"format": *formatParam},
			Paths: &spec.Paths{
				Paths: map[string]spec.PathItem{
					"/":      pi,
					"/items": pi2,
				},
			},
		},
	}
	analyzer := New(spec)

	assert.Len(t, analyzer.consumes, 2)
	assert.Len(t, analyzer.produces, 2)
	assert.Len(t, analyzer.operations, 1)
	assert.Equal(t, analyzer.operations["GET"]["/"], spec.Paths.Paths["/"].Get)

	expected := []string{"application/x-yaml"}
	sort.Sort(sort.StringSlice(expected))
	consumes := analyzer.ConsumesFor(spec.Paths.Paths["/"].Get)
	sort.Sort(sort.StringSlice(consumes))
	assert.Equal(t, expected, consumes)

	produces := analyzer.ProducesFor(spec.Paths.Paths["/"].Get)
	sort.Sort(sort.StringSlice(produces))
	assert.Equal(t, expected, produces)

	expected = []string{"application/json"}
	sort.Sort(sort.StringSlice(expected))
	consumes = analyzer.ConsumesFor(spec.Paths.Paths["/items"].Get)
	sort.Sort(sort.StringSlice(consumes))
	assert.Equal(t, expected, consumes)

	produces = analyzer.ProducesFor(spec.Paths.Paths["/items"].Get)
	sort.Sort(sort.StringSlice(produces))
	assert.Equal(t, expected, produces)

	expectedSchemes := []SecurityRequirement{SecurityRequirement{"oauth2", []string{}}, SecurityRequirement{"basic", nil}}
	schemes := analyzer.SecurityRequirementsFor(spec.Paths.Paths["/"].Get)
	assert.Equal(t, schemeNames(expectedSchemes), schemeNames(schemes))

	securityDefinitions := analyzer.SecurityDefinitionsFor(spec.Paths.Paths["/"].Get)
	assert.Equal(t, securityDefinitions["basic"], *spec.SecurityDefinitions["basic"])
	assert.Equal(t, securityDefinitions["oauth2"], *spec.SecurityDefinitions["oauth2"])

	parameters := analyzer.ParamsFor("GET", "/")
	assert.Len(t, parameters, 2)

	operations := analyzer.OperationIDs()
	assert.Len(t, operations, 2)

	producers := analyzer.RequiredProduces()
	assert.Len(t, producers, 2)
	consumers := analyzer.RequiredConsumes()
	assert.Len(t, consumers, 2)
	authSchemes := analyzer.RequiredSecuritySchemes()
	assert.Len(t, authSchemes, 3)

	ops := analyzer.Operations()
	assert.Len(t, ops, 1)
	assert.Len(t, ops["GET"], 2)

	op, ok := analyzer.OperationFor("get", "/")
	assert.True(t, ok)
	assert.NotNil(t, op)

	op, ok = analyzer.OperationFor("delete", "/")
	assert.False(t, ok)
	assert.Nil(t, op)
}

func TestDefinitionAnalysis(t *testing.T) {
	doc, err := loadSpec(filepath.Join("fixtures", "definitions.yml"))
	if assert.NoError(t, err) {
		analyzer := New(doc)
		definitions := analyzer.allSchemas
		// parameters
		assertSchemaRefExists(t, definitions, "#/parameters/someParam/schema")
		assertSchemaRefExists(t, definitions, "#/paths/~1some~1where~1{id}/parameters/1/schema")
		assertSchemaRefExists(t, definitions, "#/paths/~1some~1where~1{id}/get/parameters/1/schema")
		// responses
		assertSchemaRefExists(t, definitions, "#/responses/someResponse/schema")
		assertSchemaRefExists(t, definitions, "#/paths/~1some~1where~1{id}/get/responses/default/schema")
		assertSchemaRefExists(t, definitions, "#/paths/~1some~1where~1{id}/get/responses/200/schema")
		// definitions
		assertSchemaRefExists(t, definitions, "#/definitions/tag")
		assertSchemaRefExists(t, definitions, "#/definitions/tag/properties/id")
		assertSchemaRefExists(t, definitions, "#/definitions/tag/properties/value")
		assertSchemaRefExists(t, definitions, "#/definitions/tag/definitions/category")
		assertSchemaRefExists(t, definitions, "#/definitions/tag/definitions/category/properties/id")
		assertSchemaRefExists(t, definitions, "#/definitions/tag/definitions/category/properties/value")
		assertSchemaRefExists(t, definitions, "#/definitions/withAdditionalProps")
		assertSchemaRefExists(t, definitions, "#/definitions/withAdditionalProps/additionalProperties")
		assertSchemaRefExists(t, definitions, "#/definitions/withAdditionalItems")
		assertSchemaRefExists(t, definitions, "#/definitions/withAdditionalItems/items/0")
		assertSchemaRefExists(t, definitions, "#/definitions/withAdditionalItems/items/1")
		assertSchemaRefExists(t, definitions, "#/definitions/withAdditionalItems/additionalItems")
		assertSchemaRefExists(t, definitions, "#/definitions/withNot")
		assertSchemaRefExists(t, definitions, "#/definitions/withNot/not")
		assertSchemaRefExists(t, definitions, "#/definitions/withAnyOf")
		assertSchemaRefExists(t, definitions, "#/definitions/withAnyOf/anyOf/0")
		assertSchemaRefExists(t, definitions, "#/definitions/withAnyOf/anyOf/1")
		assertSchemaRefExists(t, definitions, "#/definitions/withAllOf")
		assertSchemaRefExists(t, definitions, "#/definitions/withAllOf/allOf/0")
		assertSchemaRefExists(t, definitions, "#/definitions/withAllOf/allOf/1")
		allOfs := analyzer.allOfs
		assert.Len(t, allOfs, 1)
		_, hasAllOf := allOfs["#/definitions/withAllOf"]
		assert.True(t, hasAllOf)
	}
}

func loadSpec(path string) (*spec.Swagger, error) {
	data, err := fmts.YAMLDoc(path)
	if err != nil {
		return nil, err
	}

	var sw spec.Swagger
	if err := json.Unmarshal(data, &sw); err != nil {
		return nil, err
	}
	return &sw, nil
}

func TestReferenceAnalysis(t *testing.T) {
	doc, err := loadSpec(filepath.Join("fixtures", "references.yml"))
	if assert.NoError(t, err) {
		definitions := New(doc).references

		// parameters
		assertRefExists(t, definitions.parameters, "#/paths/~1some~1where~1{id}/parameters/0")
		assertRefExists(t, definitions.parameters, "#/paths/~1some~1where~1{id}/get/parameters/0")

		// responses
		assertRefExists(t, definitions.responses, "#/paths/~1some~1where~1{id}/get/responses/404")

		// definitions
		assertRefExists(t, definitions.schemas, "#/responses/notFound/schema")
		assertRefExists(t, definitions.schemas, "#/paths/~1some~1where~1{id}/get/responses/200/schema")
		assertRefExists(t, definitions.schemas, "#/definitions/tag/properties/audit")

		// items
		assertRefExists(t, definitions.allRefs, "#/paths/~1some~1where~1{id}/get/parameters/1/items")
	}
}

func assertRefExists(t testing.TB, data map[string]spec.Ref, key string) bool {
	if _, ok := data[key]; !ok {
		return assert.Fail(t, fmt.Sprintf("expected %q to exist in the ref bag", key))
	}
	return true
}

func assertSchemaRefExists(t testing.TB, data map[string]SchemaRef, key string) bool {
	if _, ok := data[key]; !ok {
		return assert.Fail(t, fmt.Sprintf("expected %q to exist in schema ref bag", key))
	}
	return true
}
