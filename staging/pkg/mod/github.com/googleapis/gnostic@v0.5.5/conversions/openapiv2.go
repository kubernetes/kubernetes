// Copyright 2019 Google LLC. All Rights Reserved.
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

package conversions

import (
	"log"
	"net/url"

	openapi2 "github.com/googleapis/gnostic/openapiv2"
	discovery "github.com/googleapis/gnostic/discovery"
)

func addOpenAPI2SchemaForSchema(d *openapi2.Document, name string, schema *discovery.Schema) {
	//log.Printf("SCHEMA %s\n", name)
	d.Definitions.AdditionalProperties = append(d.Definitions.AdditionalProperties,
		&openapi2.NamedSchema{
			Name:  name,
			Value: buildOpenAPI2SchemaForSchema(schema),
		})
}

func buildOpenAPI2SchemaForSchema(schema *discovery.Schema) *openapi2.Schema {
	s := &openapi2.Schema{}

	if description := schema.Description; description != "" {
		s.Description = description
	}
	if typeName := schema.Type; typeName != "" {
		s.Type = &openapi2.TypeItem{Value: []string{typeName}}
	}
	if ref := schema.XRef; ref != "" {
		s.XRef = "#/definitions/" + ref
	}
	if len(schema.Enum) > 0 {
		for _, e := range schema.Enum {
			s.Enum = append(s.Enum, &openapi2.Any{Yaml: e})
		}
	}
	if schema.Items != nil {
		s2 := buildOpenAPI2SchemaForSchema(schema.Items)
		s.Items = &openapi2.ItemsItem{}
		s.Items.Schema = append(s.Items.Schema, s2)
	}
	if schema.Properties != nil {
		if len(schema.Properties.AdditionalProperties) > 0 {
			s.Properties = &openapi2.Properties{}
			for _, pair := range schema.Properties.AdditionalProperties {
				s.Properties.AdditionalProperties = append(s.Properties.AdditionalProperties,
					&openapi2.NamedSchema{
						Name:  pair.Name,
						Value: buildOpenAPI2SchemaForSchema(pair.Value),
					},
				)
			}
		}
	}
	// assume that all schemas are closed
	s.AdditionalProperties = &openapi2.AdditionalPropertiesItem{Oneof: &openapi2.AdditionalPropertiesItem_Boolean{Boolean: false}}
	return s
}

func buildOpenAPI2ParameterForParameter(name string, p *discovery.Parameter) *openapi2.Parameter {
	//log.Printf("- PARAMETER %+v\n", p.Name)
	typeName := p.Type
	format := p.Format
	location := p.Location
	switch location {
	case "query":
		return &openapi2.Parameter{
			Oneof: &openapi2.Parameter_NonBodyParameter{
				NonBodyParameter: &openapi2.NonBodyParameter{
					Oneof: &openapi2.NonBodyParameter_QueryParameterSubSchema{
						QueryParameterSubSchema: &openapi2.QueryParameterSubSchema{
							Name:        name,
							In:          "query",
							Description: p.Description,
							Required:    p.Required,
							Type:        typeName,
							Format:      format,
						},
					},
				},
			},
		}
	case "path":
		return &openapi2.Parameter{
			Oneof: &openapi2.Parameter_NonBodyParameter{
				NonBodyParameter: &openapi2.NonBodyParameter{
					Oneof: &openapi2.NonBodyParameter_PathParameterSubSchema{
						PathParameterSubSchema: &openapi2.PathParameterSubSchema{
							Name:        name,
							In:          "path",
							Description: p.Description,
							Required:    p.Required,
							Type:        typeName,
							Format:      format,
						},
					},
				},
			},
		}
	default:
		return nil
	}
}

func buildOpenAPI2ParameterForRequest(p *discovery.Request) *openapi2.Parameter {
	return &openapi2.Parameter{
		Oneof: &openapi2.Parameter_BodyParameter{
			BodyParameter: &openapi2.BodyParameter{
				Name:        "resource",
				In:          "body",
				Description: "",
				Schema:      &openapi2.Schema{XRef: "#/definitions/" + p.XRef},
			},
		},
	}
}

func buildOpenAPI2ResponseForResponse(response *discovery.Response) *openapi2.Response {
	//log.Printf("- RESPONSE %+v\n", schema)
	if response == nil {
		return &openapi2.Response{
			Description: "Successful operation",
		}
	}
	ref := response.XRef
	if ref == "" {
		log.Printf("WARNING: Unhandled response %+v", response)
	}
	return &openapi2.Response{
		Description: "Successful operation",
		Schema: &openapi2.SchemaItem{
			Oneof: &openapi2.SchemaItem_Schema{
				Schema: &openapi2.Schema{
					XRef: "#/definitions/" + ref,
				},
			},
		},
	}
}

func buildOpenAPI2OperationForMethod(method *discovery.Method) *openapi2.Operation {
	//log.Printf("METHOD %s %s %s %s\n", method.Name, method.path(), method.HTTPMethod, method.ID)
	//log.Printf("MAP %+v\n", method.JSONMap)
	parameters := make([]*openapi2.ParametersItem, 0)
	if method.Parameters != nil {
		for _, pair := range method.Parameters.AdditionalProperties {
			parameters = append(parameters, &openapi2.ParametersItem{
				Oneof: &openapi2.ParametersItem_Parameter{
					Parameter: buildOpenAPI2ParameterForParameter(pair.Name, pair.Value),
				},
			})
		}
	}
	responses := &openapi2.Responses{
		ResponseCode: []*openapi2.NamedResponseValue{
			&openapi2.NamedResponseValue{
				Name: "default",
				Value: &openapi2.ResponseValue{
					Oneof: &openapi2.ResponseValue_Response{
						Response: buildOpenAPI2ResponseForResponse(method.Response),
					},
				},
			},
		},
	}
	if method.Request != nil {
		parameter := buildOpenAPI2ParameterForRequest(method.Request)
		parameters = append(parameters, &openapi2.ParametersItem{
			Oneof: &openapi2.ParametersItem_Parameter{
				Parameter: parameter,
			},
		})
	}
	return &openapi2.Operation{
		Description: method.Description,
		OperationId: method.Id,
		Parameters:  parameters,
		Responses:   responses,
	}
}

func getOpenAPI2PathItemForPath(d *openapi2.Document, path string) *openapi2.PathItem {
	// First, try to find a path item with the specified path. If it exists, return it.
	for _, item := range d.Paths.Path {
		if item.Name == path {
			return item.Value
		}
	}
	// Otherwise, create and return a new path item.
	pathItem := &openapi2.PathItem{}
	d.Paths.Path = append(d.Paths.Path,
		&openapi2.NamedPathItem{
			Name:  path,
			Value: pathItem,
		},
	)
	return pathItem
}

func addOpenAPI2PathsForMethod(d *openapi2.Document, name string, method *discovery.Method) {
	operation := buildOpenAPI2OperationForMethod(method)
	pathItem := getOpenAPI2PathItemForPath(d, pathForMethod(method.Path))
	switch method.HttpMethod {
	case "GET":
		pathItem.Get = operation
	case "POST":
		pathItem.Post = operation
	case "PUT":
		pathItem.Put = operation
	case "DELETE":
		pathItem.Delete = operation
	case "PATCH":
		pathItem.Patch = operation
	default:
		log.Printf("WARNING: Unknown HTTP method %s", method.HttpMethod)
	}
}

func addOpenAPI2PathsForResource(d *openapi2.Document, name string, resource *discovery.Resource) {
	//log.Printf("RESOURCE %s (%s)\n", resource.Name, resource.FullName)
	if resource.Methods != nil {
		for _, pair := range resource.Methods.AdditionalProperties {
			addOpenAPI2PathsForMethod(d, pair.Name, pair.Value)
		}
	}
	if resource.Resources != nil {
		for _, pair := range resource.Resources.AdditionalProperties {
			addOpenAPI2PathsForResource(d, pair.Name, pair.Value)
		}
	}
}

func removeTrailingSlash(path string) string {
	if len(path) > 1 && path[len(path)-1] == '/' {
		return path[0 : len(path)-1]
	}
	return path
}

// OpenAPIv2 returns an OpenAPI v2 representation of this Discovery document
func OpenAPIv2(api *discovery.Document) (*openapi2.Document, error) {
	d := &openapi2.Document{}
	d.Swagger = "2.0"
	d.Info = &openapi2.Info{
		Title:       api.Title,
		Version:     api.Version,
		Description: api.Description,
	}
	url, _ := url.Parse(api.RootUrl)
	d.Host = url.Host
	d.BasePath = removeTrailingSlash(api.BasePath)
	d.Schemes = []string{url.Scheme}
	d.Consumes = []string{"application/json"}
	d.Produces = []string{"application/json"}
	d.Paths = &openapi2.Paths{}
	d.Definitions = &openapi2.Definitions{}
	if api.Schemas != nil {
		for _, pair := range api.Schemas.AdditionalProperties {
			addOpenAPI2SchemaForSchema(d, pair.Name, pair.Value)
		}
	}
	if api.Methods != nil {
		for _, pair := range api.Methods.AdditionalProperties {
			addOpenAPI2PathsForMethod(d, pair.Name, pair.Value)
		}
	}
	if api.Resources != nil {
		for _, pair := range api.Resources.AdditionalProperties {
			addOpenAPI2PathsForResource(d, pair.Name, pair.Value)
		}
	}
	return d, nil
}
