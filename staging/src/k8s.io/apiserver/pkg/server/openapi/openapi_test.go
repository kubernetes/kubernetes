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
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"testing"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/openapi"
)

// setUp is a convenience function for setting up for (most) tests.
func setUp(t *testing.T, fullMethods bool) (openAPI, *restful.Container, *assert.Assertions) {
	assert := assert.New(t)
	config, container := getConfig(fullMethods)
	return openAPI{
		config: config,
		swagger: &spec.Swagger{
			SwaggerProps: spec.SwaggerProps{
				Swagger:     OpenAPIVersion,
				Definitions: spec.Definitions{},
				Paths:       &spec.Paths{Paths: map[string]spec.PathItem{}},
				Info:        config.Info,
			},
		},
	}, container, assert
}

func noOp(request *restful.Request, response *restful.Response) {}

// Test input
type TestInput struct {
	// Name of the input
	Name string `json:"name,omitempty"`
	// ID of the input
	ID   int      `json:"id,omitempty"`
	Tags []string `json:"tags,omitempty"`
}

// Test output
type TestOutput struct {
	// Name of the output
	Name string `json:"name,omitempty"`
	// Number of outputs
	Count int `json:"count,omitempty"`
}

func (_ TestInput) OpenAPIDefinition() *openapi.OpenAPIDefinition {
	schema := spec.Schema{}
	schema.Description = "Test input"
	schema.Properties = map[string]spec.Schema{
		"name": {
			SchemaProps: spec.SchemaProps{
				Description: "Name of the input",
				Type:        []string{"string"},
				Format:      "",
			},
		},
		"id": {
			SchemaProps: spec.SchemaProps{
				Description: "ID of the input",
				Type:        []string{"integer"},
				Format:      "int32",
			},
		},
		"tags": {
			SchemaProps: spec.SchemaProps{
				Description: "",
				Type:        []string{"array"},
				Items: &spec.SchemaOrArray{
					Schema: &spec.Schema{
						SchemaProps: spec.SchemaProps{
							Type:   []string{"string"},
							Format: "",
						},
					},
				},
			},
		},
	}
	schema.Extensions = spec.Extensions{"x-test": "test"}
	return &openapi.OpenAPIDefinition{
		Schema:       schema,
		Dependencies: []string{},
	}
}

func (_ TestOutput) OpenAPIDefinition() *openapi.OpenAPIDefinition {
	schema := spec.Schema{}
	schema.Description = "Test output"
	schema.Properties = map[string]spec.Schema{
		"name": {
			SchemaProps: spec.SchemaProps{
				Description: "Name of the output",
				Type:        []string{"string"},
				Format:      "",
			},
		},
		"count": {
			SchemaProps: spec.SchemaProps{
				Description: "Number of outputs",
				Type:        []string{"integer"},
				Format:      "int32",
			},
		},
	}
	return &openapi.OpenAPIDefinition{
		Schema:       schema,
		Dependencies: []string{},
	}
}

var _ openapi.OpenAPIDefinitionGetter = TestInput{}
var _ openapi.OpenAPIDefinitionGetter = TestOutput{}

func getTestRoute(ws *restful.WebService, method string, additionalParams bool, opPrefix string) *restful.RouteBuilder {
	ret := ws.Method(method).
		Path("/test/{path:*}").
		Doc(fmt.Sprintf("%s test input", method)).
		Operation(fmt.Sprintf("%s%sTestInput", method, opPrefix)).
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

func getConfig(fullMethods bool) (*openapi.Config, *restful.Container) {
	mux := http.NewServeMux()
	container := restful.NewContainer()
	container.ServeMux = mux
	ws := new(restful.WebService)
	ws.Path("/foo")
	ws.Route(getTestRoute(ws, "get", true, "foo"))
	if fullMethods {
		ws.Route(getTestRoute(ws, "post", false, "foo")).
			Route(getTestRoute(ws, "put", false, "foo")).
			Route(getTestRoute(ws, "head", false, "foo")).
			Route(getTestRoute(ws, "patch", false, "foo")).
			Route(getTestRoute(ws, "options", false, "foo")).
			Route(getTestRoute(ws, "delete", false, "foo"))

	}
	ws.Path("/bar")
	ws.Route(getTestRoute(ws, "get", true, "bar"))
	if fullMethods {
		ws.Route(getTestRoute(ws, "post", false, "bar")).
			Route(getTestRoute(ws, "put", false, "bar")).
			Route(getTestRoute(ws, "head", false, "bar")).
			Route(getTestRoute(ws, "patch", false, "bar")).
			Route(getTestRoute(ws, "options", false, "bar")).
			Route(getTestRoute(ws, "delete", false, "bar"))

	}
	container.Add(ws)
	return &openapi.Config{
		ProtocolList: []string{"https"},
		Info: &spec.Info{
			InfoProps: spec.InfoProps{
				Title:       "TestAPI",
				Description: "Test API",
				Version:     "unversioned",
			},
		},
		GetDefinitions: func(_ openapi.ReferenceCallback) map[string]openapi.OpenAPIDefinition {
			return map[string]openapi.OpenAPIDefinition{
				"k8s.io/apiserver/pkg/server/openapi.TestInput":  *TestInput{}.OpenAPIDefinition(),
				"k8s.io/apiserver/pkg/server/openapi.TestOutput": *TestOutput{}.OpenAPIDefinition(),
				// Bazel changes the package name, this is ok for testing, but we need to fix it if it happened
				// in the main code.
				"k8s.io/apiserver/pkg/server/openapi/go_default_test.TestInput":  *TestInput{}.OpenAPIDefinition(),
				"k8s.io/apiserver/pkg/server/openapi/go_default_test.TestOutput": *TestOutput{}.OpenAPIDefinition(),
			}
		},
		GetDefinitionName: func(name string) (string, spec.Extensions) {
			friendlyName := name[strings.LastIndex(name, "/")+1:]
			if strings.HasPrefix(friendlyName, "go_default_test") {
				friendlyName = "openapi" + friendlyName[len("go_default_test"):]
			}
			return friendlyName, spec.Extensions{"x-test2": "test2"}
		},
	}, container
}

func getTestOperation(method string, opPrefix string) *spec.Operation {
	return &spec.Operation{
		OperationProps: spec.OperationProps{
			Description: fmt.Sprintf("%s test input", method),
			Consumes:    []string{"application/json"},
			Produces:    []string{"application/json"},
			Schemes:     []string{"https"},
			Parameters:  []spec.Parameter{},
			Responses:   getTestResponses(),
			ID:          fmt.Sprintf("%s%sTestInput", method, opPrefix),
		},
	}
}

func getTestPathItem(allMethods bool, opPrefix string) spec.PathItem {
	ret := spec.PathItem{
		PathItemProps: spec.PathItemProps{
			Get:        getTestOperation("get", opPrefix),
			Parameters: getTestCommonParameters(),
		},
	}
	ret.Get.Parameters = getAdditionalTestParameters()
	if allMethods {
		ret.Put = getTestOperation("put", opPrefix)
		ret.Put.Parameters = getTestParameters()
		ret.Post = getTestOperation("post", opPrefix)
		ret.Post.Parameters = getTestParameters()
		ret.Head = getTestOperation("head", opPrefix)
		ret.Head.Parameters = getTestParameters()
		ret.Patch = getTestOperation("patch", opPrefix)
		ret.Patch.Parameters = getTestParameters()
		ret.Delete = getTestOperation("delete", opPrefix)
		ret.Delete.Parameters = getTestParameters()
		ret.Options = getTestOperation("options", opPrefix)
		ret.Options.Parameters = getTestParameters()
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
	ret := make([]spec.Parameter, 2)
	ret[0] = spec.Parameter{
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
	ret[1] = spec.Parameter{
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

func getTestParameters() []spec.Parameter {
	ret := make([]spec.Parameter, 1)
	ret[0] = spec.Parameter{
		ParamProps: spec.ParamProps{
			Name:     "body",
			In:       "body",
			Required: true,
			Schema:   getRefSchema("#/definitions/openapi.TestInput"),
		},
	}
	return ret
}

func getAdditionalTestParameters() []spec.Parameter {
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
		ParamProps: spec.ParamProps{
			Name:        "fparam",
			Description: "a test form parameter",
			In:          "formData",
		},
		SimpleSchema: spec.SimpleSchema{
			Type: "number",
		},
		CommonValidations: spec.CommonValidations{
			UniqueItems: true,
		},
	}
	ret[2] = spec.Parameter{
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

func getTestInputDefinition() spec.Schema {
	return spec.Schema{
		SchemaProps: spec.SchemaProps{
			Description: "Test input",
			Properties: map[string]spec.Schema{
				"id": {
					SchemaProps: spec.SchemaProps{
						Description: "ID of the input",
						Type:        spec.StringOrArray{"integer"},
						Format:      "int32",
					},
				},
				"name": {
					SchemaProps: spec.SchemaProps{
						Description: "Name of the input",
						Type:        spec.StringOrArray{"string"},
					},
				},
				"tags": {
					SchemaProps: spec.SchemaProps{
						Type: spec.StringOrArray{"array"},
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
		VendorExtensible: spec.VendorExtensible{
			Extensions: spec.Extensions{
				"x-test":  "test",
				"x-test2": "test2",
			},
		},
	}
}

func getTestOutputDefinition() spec.Schema {
	return spec.Schema{
		SchemaProps: spec.SchemaProps{
			Description: "Test output",
			Properties: map[string]spec.Schema{
				"count": {
					SchemaProps: spec.SchemaProps{
						Description: "Number of outputs",
						Type:        spec.StringOrArray{"integer"},
						Format:      "int32",
					},
				},
				"name": {
					SchemaProps: spec.SchemaProps{
						Description: "Name of the output",
						Type:        spec.StringOrArray{"string"},
					},
				},
			},
		},
		VendorExtensible: spec.VendorExtensible{
			Extensions: spec.Extensions{
				"x-test2": "test2",
			},
		},
	}
}

func TestBuildSwaggerSpec(t *testing.T) {
	o, container, assert := setUp(t, true)
	expected := &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Info: &spec.Info{
				InfoProps: spec.InfoProps{
					Title:       "TestAPI",
					Description: "Test API",
					Version:     "unversioned",
				},
			},
			Swagger: "2.0",
			Paths: &spec.Paths{
				Paths: map[string]spec.PathItem{
					"/foo/test/{path}": getTestPathItem(true, "foo"),
					"/bar/test/{path}": getTestPathItem(true, "bar"),
				},
			},
			Definitions: spec.Definitions{
				"openapi.TestInput":  getTestInputDefinition(),
				"openapi.TestOutput": getTestOutputDefinition(),
			},
		},
	}
	err := o.init(container.RegisteredWebServices())
	if !assert.NoError(err) {
		return
	}
	expected_json, err := json.Marshal(expected)
	if !assert.NoError(err) {
		return
	}
	actual_json, err := json.Marshal(o.swagger)
	if !assert.NoError(err) {
		return
	}
	assert.Equal(string(expected_json), string(actual_json))
}
