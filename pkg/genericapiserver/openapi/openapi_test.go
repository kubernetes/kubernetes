/*
Copyright 2016 The Kubernetes Authors.

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

package openapi

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
	"sort"
)

// setUp is a convenience function for setting up for (most) tests.
func setUp(t *testing.T, fullMethods bool) (openAPI, *assert.Assertions) {
	assert := assert.New(t)
	config := Config{
		SwaggerConfig: getSwaggerConfig(fullMethods),
		Info: &spec.Info{
			InfoProps: spec.InfoProps{
				Title:       "TestAPI",
				Description: "Test API",
			},
		},
	}
	return openAPI{config: &config}, assert
}

func noOp(request *restful.Request, response *restful.Response) {}

type TestInput struct {
	Name string   `json:"name,omitempty"`
	ID   int      `json:"id,omitempty"`
	Tags []string `json:"tags,omitempty"`
}

type TestOutput struct {
	Name  string `json:"name,omitempty"`
	Count int    `json:"count,omitempty"`
}

func (t TestInput) SwaggerDoc() map[string]string {
	return map[string]string{
		"":     "Test input",
		"name": "Name of the input",
		"id":   "ID of the input",
	}
}

func (t TestOutput) SwaggerDoc() map[string]string {
	return map[string]string{
		"":      "Test output",
		"name":  "Name of the output",
		"count": "Number of outputs",
	}
}

func getTestRoute(ws *restful.WebService, method string, additionalParams bool) *restful.RouteBuilder {
	ret := ws.Method(method).
		Path("/test/{path:*}").
		Doc(fmt.Sprintf("%s test input", method)).
		Operation(fmt.Sprintf("%sTestInput", method)).
		Produces(restful.MIME_JSON).
		Consumes(restful.MIME_JSON).
		Param(ws.PathParameter("path", "path to the resource").DataType("string")).
		Param(ws.QueryParameter("pretty", "If 'true', then the output is pretty printed.")).
		Reads(TestInput{}).
		Returns(200, "OK", TestOutput{}).
		Writes(TestOutput{}).
		To(noOp)
	if additionalParams {
		ret.Param(ws.HeaderParameter("hparam", "a test head parameter").DataType("integer"))
		ret.Param(ws.FormParameter("fparam", "a test form parameter").DataType("number"))
	}
	return ret
}

func getSwaggerConfig(fullMethods bool) *swagger.Config {
	mux := http.NewServeMux()
	container := restful.NewContainer()
	container.ServeMux = mux
	ws := new(restful.WebService)
	ws.Path("/foo")
	ws.Route(getTestRoute(ws, "get", true))
	if fullMethods {
		ws.Route(getTestRoute(ws, "post", false)).
			Route(getTestRoute(ws, "put", false)).
			Route(getTestRoute(ws, "head", false)).
			Route(getTestRoute(ws, "patch", false)).
			Route(getTestRoute(ws, "options", false)).
			Route(getTestRoute(ws, "delete", false))

	}
	ws.Path("/bar")
	ws.Route(getTestRoute(ws, "get", true))
	if fullMethods {
		ws.Route(getTestRoute(ws, "post", false)).
			Route(getTestRoute(ws, "put", false)).
			Route(getTestRoute(ws, "head", false)).
			Route(getTestRoute(ws, "patch", false)).
			Route(getTestRoute(ws, "options", false)).
			Route(getTestRoute(ws, "delete", false))

	}
	container.Add(ws)
	return &swagger.Config{
		WebServicesUrl: "https://test-server",
		WebServices:    container.RegisteredWebServices(),
	}
}

func getTestOperation(method string) *spec.Operation {
	return &spec.Operation{
		OperationProps: spec.OperationProps{
			Description: fmt.Sprintf("%s test input", method),
			Consumes:    []string{"application/json"},
			Produces:    []string{"application/json"},
			Schemes:     []string{"https"},
			Parameters:  []spec.Parameter{},
			Responses:   getTestResponses(),
		},
	}
}

func getTestPathItem(allMethods bool) spec.PathItem {
	ret := spec.PathItem{
		PathItemProps: spec.PathItemProps{
			Get:        getTestOperation("get"),
			Parameters: getTestCommonParameters(),
		},
	}
	ret.Get.Parameters = getAdditionalTestParameters()
	if allMethods {
		ret.PathItemProps.Put = getTestOperation("put")
		ret.PathItemProps.Post = getTestOperation("post")
		ret.PathItemProps.Head = getTestOperation("head")
		ret.PathItemProps.Patch = getTestOperation("patch")
		ret.PathItemProps.Delete = getTestOperation("delete")
		ret.PathItemProps.Options = getTestOperation("options")
	}
	return ret
}

func getRefSchema(ref string) *spec.Schema {
	return &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Ref: spec.MustCreateRef(ref),
		},
	}
}

func getTestResponses() *spec.Responses {
	ret := spec.Responses{
		ResponsesProps: spec.ResponsesProps{
			StatusCodeResponses: map[int]spec.Response{},
		},
	}
	ret.StatusCodeResponses[200] = spec.Response{
		ResponseProps: spec.ResponseProps{
			Description: "OK",
			Schema:      getRefSchema("#/definitions/openapi.TestOutput"),
		},
	}
	return &ret
}

func getTestCommonParameters() []spec.Parameter {
	ret := make([]spec.Parameter, 3)
	ret[0] = spec.Parameter{
		ParamProps: spec.ParamProps{
			Name:     "body",
			In:       "body",
			Required: true,
			Schema:   getRefSchema("#/definitions/openapi.TestInput"),
		},
	}
	ret[1] = spec.Parameter{
		SimpleSchema: spec.SimpleSchema{
			Type: "string",
		},
		ParamProps: spec.ParamProps{
			Description: "path to the resource",
			Name:        "path",
			In:          "path",
			Required:    true,
		},
		CommonValidations: spec.CommonValidations{
			UniqueItems: true,
		},
	}
	ret[2] = spec.Parameter{
		SimpleSchema: spec.SimpleSchema{
			Type: "string",
		},
		ParamProps: spec.ParamProps{
			Description: "If 'true', then the output is pretty printed.",
			Name:        "pretty",
			In:          "query",
		},
		CommonValidations: spec.CommonValidations{
			UniqueItems: true,
		},
	}
	return ret
}

func getAdditionalTestParameters() []spec.Parameter {
	ret := make([]spec.Parameter, 2)
	ret[0] = spec.Parameter{
		ParamProps: spec.ParamProps{
			Name:        "fparam",
			Description: "a test form parameter",
			In:          "form",
		},
		SimpleSchema: spec.SimpleSchema{
			Type: "number",
		},
		CommonValidations: spec.CommonValidations{
			UniqueItems: true,
		},
	}
	ret[1] = spec.Parameter{
		SimpleSchema: spec.SimpleSchema{
			Type: "integer",
		},
		ParamProps: spec.ParamProps{
			Description: "a test head parameter",
			Name:        "hparam",
			In:          "header",
		},
		CommonValidations: spec.CommonValidations{
			UniqueItems: true,
		},
	}
	return ret
}

type Parameters []spec.Parameter

func (s Parameters) Len() int      { return len(s) }
func (s Parameters) Swap(i, j int) { s[i], s[j] = s[j], s[i] }

type ByName struct {
	Parameters
}

func (s ByName) Less(i, j int) bool {
	return s.Parameters[i].Name < s.Parameters[j].Name
}

// TODO(mehdy): Consider sort parameters in actual spec generation for more predictable spec generation
func sortParameters(s *spec.Swagger) *spec.Swagger {
	for k, p := range s.Paths.Paths {
		sort.Sort(ByName{p.Parameters})
		sort.Sort(ByName{p.Get.Parameters})
		sort.Sort(ByName{p.Put.Parameters})
		sort.Sort(ByName{p.Post.Parameters})
		sort.Sort(ByName{p.Head.Parameters})
		sort.Sort(ByName{p.Delete.Parameters})
		sort.Sort(ByName{p.Options.Parameters})
		sort.Sort(ByName{p.Patch.Parameters})
		s.Paths.Paths[k] = p // Unnecessary?! Magic!!!
	}
	return s
}

func getTestInputDefinition() spec.Schema {
	return spec.Schema{
		SchemaProps: spec.SchemaProps{
			Description: "Test input",
			Required:    []string{},
			Properties: map[string]spec.Schema{
				"id": {
					SchemaProps: spec.SchemaProps{
						Description: "ID of the input",
						Type:        spec.StringOrArray{"integer"},
						Format:      "int32",
						Enum:        []interface{}{},
					},
				},
				"name": {
					SchemaProps: spec.SchemaProps{
						Description: "Name of the input",
						Type:        spec.StringOrArray{"string"},
						Enum:        []interface{}{},
					},
				},
				"tags": {
					SchemaProps: spec.SchemaProps{
						Type: spec.StringOrArray{"array"},
						Enum: []interface{}{},
						Items: &spec.SchemaOrArray{
							Schema: &spec.Schema{
								SchemaProps: spec.SchemaProps{
									Type: spec.StringOrArray{"string"},
								},
							},
						},
					},
				},
			},
		},
	}
}

func getTestOutputDefinition() spec.Schema {
	return spec.Schema{
		SchemaProps: spec.SchemaProps{
			Description: "Test output",
			Required:    []string{},
			Properties: map[string]spec.Schema{
				"count": {
					SchemaProps: spec.SchemaProps{
						Description: "Number of outputs",
						Type:        spec.StringOrArray{"integer"},
						Format:      "int32",
						Enum:        []interface{}{},
					},
				},
				"name": {
					SchemaProps: spec.SchemaProps{
						Description: "Name of the output",
						Type:        spec.StringOrArray{"string"},
						Enum:        []interface{}{},
					},
				},
			},
		},
	}
}

func TestBuildSwaggerSpec(t *testing.T) {
	o, assert := setUp(t, true)
	expected := &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Info: &spec.Info{
				InfoProps: spec.InfoProps{
					Title:       "TestAPI",
					Description: "Test API",
				},
			},
			Swagger: "2.0",
			Paths: &spec.Paths{
				Paths: map[string]spec.PathItem{
					"/foo/test/{path}": getTestPathItem(true),
					"/bar/test/{path}": getTestPathItem(true),
				},
			},
			Definitions: spec.Definitions{
				"openapi.TestInput":  getTestInputDefinition(),
				"openapi.TestOutput": getTestOutputDefinition(),
			},
		},
	}
	err := o.buildSwaggerSpec()
	if assert.NoError(err) {
		sortParameters(expected)
		sortParameters(o.swagger)
		assert.Equal(expected, o.swagger)
	}
}

func TestBuildSwaggerSpecTwice(t *testing.T) {
	o, assert := setUp(t, true)
	err := o.buildSwaggerSpec()
	if assert.NoError(err) {
		assert.Error(o.buildSwaggerSpec(), "Swagger spec is already built. Duplicate call to buildSwaggerSpec is not allowed.")
	}

}
func TestBuildDefinitions(t *testing.T) {
	o, assert := setUp(t, true)
	expected := spec.Definitions{
		"openapi.TestInput":  getTestInputDefinition(),
		"openapi.TestOutput": getTestOutputDefinition(),
	}
	def, err := o.buildDefinitions()
	if assert.NoError(err) {
		assert.Equal(expected, def)
	}
}

func TestBuildProtocolList(t *testing.T) {
	assert := assert.New(t)
	o := openAPI{config: &Config{SwaggerConfig: &swagger.Config{WebServicesUrl: "https://something"}}}
	p, err := o.buildProtocolList()
	if assert.NoError(err) {
		assert.Equal([]string{"https"}, p)
	}
	o = openAPI{config: &Config{SwaggerConfig: &swagger.Config{WebServicesUrl: "http://something"}}}
	p, err = o.buildProtocolList()
	if assert.NoError(err) {
		assert.Equal([]string{"http"}, p)
	}
	o = openAPI{config: &Config{SwaggerConfig: &swagger.Config{WebServicesUrl: "something"}}}
	p, err = o.buildProtocolList()
	if assert.NoError(err) {
		assert.Equal([]string{"http"}, p)
	}
}
