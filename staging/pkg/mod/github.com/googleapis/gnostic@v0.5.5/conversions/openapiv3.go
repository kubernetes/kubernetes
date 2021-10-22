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
	"strings"

	openapi3 "github.com/googleapis/gnostic/openapiv3"
	discovery "github.com/googleapis/gnostic/discovery"
)

func pathForMethod(path string) string {
	return "/" + strings.Replace(path, "{+", "{", -1)
}

func addOpenAPI3SchemaForSchema(d *openapi3.Document, name string, schema *discovery.Schema) {
	d.Components.Schemas.AdditionalProperties = append(d.Components.Schemas.AdditionalProperties,
		&openapi3.NamedSchemaOrReference{
			Name:  name,
			Value: buildOpenAPI3SchemaOrReferenceForSchema(schema),
		})
}

func buildOpenAPI3SchemaOrReferenceForSchema(schema *discovery.Schema) *openapi3.SchemaOrReference {
	if ref := schema.XRef; ref != "" {
		return &openapi3.SchemaOrReference{
			Oneof: &openapi3.SchemaOrReference_Reference{
				Reference: &openapi3.Reference{
					XRef: "#/definitions/" + ref,
				},
			},
		}
	}

	s := &openapi3.Schema{}

	if description := schema.Description; description != "" {
		s.Description = description
	}
	if typeName := schema.Type; typeName != "" {
		s.Type = typeName
	}
	if len(schema.Enum) > 0 {
		for _, e := range schema.Enum {
			s.Enum = append(s.Enum, &openapi3.Any{Yaml: e})
		}
	}
	if schema.Items != nil {
		s.Items = &openapi3.ItemsItem{
			SchemaOrReference: []*openapi3.SchemaOrReference{buildOpenAPI3SchemaOrReferenceForSchema(schema.Items)},
		}
	}
	if (schema.Properties != nil) && (len(schema.Properties.AdditionalProperties) > 0) {
		s.Properties = &openapi3.Properties{}
		for _, pair := range schema.Properties.AdditionalProperties {
			s.Properties.AdditionalProperties = append(s.Properties.AdditionalProperties,
				&openapi3.NamedSchemaOrReference{
					Name:  pair.Name,
					Value: buildOpenAPI3SchemaOrReferenceForSchema(pair.Value),
				},
			)
		}
	}
	return &openapi3.SchemaOrReference{
		Oneof: &openapi3.SchemaOrReference_Schema{
			Schema: s,
		},
	}
}

func buildOpenAPI3ParameterForParameter(name string, p *discovery.Parameter) *openapi3.Parameter {
	typeName := p.Type
	format := p.Format
	location := p.Location
	switch location {
	case "query", "path":
		return &openapi3.Parameter{
			Name:        name,
			In:          location,
			Description: p.Description,
			Required:    p.Required,
			Schema: &openapi3.SchemaOrReference{
				Oneof: &openapi3.SchemaOrReference_Schema{
					Schema: &openapi3.Schema{
						Type:   typeName,
						Format: format,
					},
				},
			},
		}
	default:
		return nil
	}
}

func buildOpenAPI3RequestBodyForRequest(request *discovery.Request) *openapi3.RequestBody {
	ref := request.XRef
	if ref == "" {
		log.Printf("WARNING: Unhandled request schema %+v", request)
	}
	return &openapi3.RequestBody{
		Content: &openapi3.MediaTypes{
			AdditionalProperties: []*openapi3.NamedMediaType{
				&openapi3.NamedMediaType{
					Name: "application/json",
					Value: &openapi3.MediaType{
						Schema: &openapi3.SchemaOrReference{
							Oneof: &openapi3.SchemaOrReference_Reference{
								Reference: &openapi3.Reference{
									XRef: "#/definitions/" + ref,
								},
							},
						},
					},
				},
			},
		},
	}
}

func buildOpenAPI3ResponseForResponse(response *discovery.Response, hasDataWrapper bool) *openapi3.Response {
	if response == nil {
		return &openapi3.Response{
			Description: "Successful operation",
		}
	} else {
		ref := response.XRef
		if ref == "" {
			log.Printf("WARNING: Unhandled response %+v", response)
		}
		return &openapi3.Response{
			Description: "Successful operation",
			Content: &openapi3.MediaTypes{
				AdditionalProperties: []*openapi3.NamedMediaType{
					&openapi3.NamedMediaType{
						Name: "application/json",
						Value: &openapi3.MediaType{
							Schema: &openapi3.SchemaOrReference{
								Oneof: &openapi3.SchemaOrReference_Reference{
									Reference: &openapi3.Reference{
										XRef: "#/definitions/" + ref,
									},
								},
							},
						},
					},
				},
			},
		}
	}
}

func buildOpenAPI3OperationForMethod(method *discovery.Method, hasDataWrapper bool) *openapi3.Operation {
	if method == nil {
		return nil
	}
	parameters := make([]*openapi3.ParameterOrReference, 0)
	if method.Parameters != nil {
		for _, pair := range method.Parameters.AdditionalProperties {
			parameters = append(parameters, &openapi3.ParameterOrReference{
				Oneof: &openapi3.ParameterOrReference_Parameter{
					Parameter: buildOpenAPI3ParameterForParameter(pair.Name, pair.Value),
				},
			})
		}
	}
	responses := &openapi3.Responses{
		ResponseOrReference: []*openapi3.NamedResponseOrReference{
			&openapi3.NamedResponseOrReference{
				Name: "default",
				Value: &openapi3.ResponseOrReference{
					Oneof: &openapi3.ResponseOrReference_Response{
						Response: buildOpenAPI3ResponseForResponse(method.Response, hasDataWrapper),
					},
				},
			},
		},
	}
	var requestBodyOrReference *openapi3.RequestBodyOrReference
	if method.Request != nil {
		requestBody := buildOpenAPI3RequestBodyForRequest(method.Request)
		requestBodyOrReference = &openapi3.RequestBodyOrReference{
			Oneof: &openapi3.RequestBodyOrReference_RequestBody{
				RequestBody: requestBody,
			},
		}
	}
	return &openapi3.Operation{
		Description: method.Description,
		OperationId: method.Id,
		Parameters:  parameters,
		Responses:   responses,
		RequestBody: requestBodyOrReference,
	}
}

func getOpenAPI3PathItemForPath(d *openapi3.Document, path string) *openapi3.PathItem {
	// First, try to find a path item with the specified path. If it exists, return it.
	for _, item := range d.Paths.Path {
		if item.Name == path {
			return item.Value
		}
	}
	// Otherwise, create and return a new path item.
	pathItem := &openapi3.PathItem{}
	d.Paths.Path = append(d.Paths.Path,
		&openapi3.NamedPathItem{
			Name:  path,
			Value: pathItem,
		},
	)
	return pathItem
}

func addOpenAPI3PathsForMethod(d *openapi3.Document, name string, method *discovery.Method, hasDataWrapper bool) {
	operation := buildOpenAPI3OperationForMethod(method, hasDataWrapper)
	pathItem := getOpenAPI3PathItemForPath(d, pathForMethod(method.Path))
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

func addOpenAPI3PathsForResource(d *openapi3.Document, resource *discovery.Resource, hasDataWrapper bool) {
	if resource.Methods != nil {
		for _, pair := range resource.Methods.AdditionalProperties {
			addOpenAPI3PathsForMethod(d, pair.Name, pair.Value, hasDataWrapper)
		}
	}
	if resource.Resources != nil {
		for _, pair := range resource.Resources.AdditionalProperties {
			addOpenAPI3PathsForResource(d, pair.Value, hasDataWrapper)
		}
	}
}

// OpenAPIv3 returns an OpenAPI v3 representation of a Discovery document
func OpenAPIv3(api *discovery.Document) (*openapi3.Document, error) {
	d := &openapi3.Document{}
	d.Openapi = "3.0"
	d.Info = &openapi3.Info{
		Title:       api.Title,
		Version:     api.Version,
		Description: api.Description,
	}
	d.Servers = make([]*openapi3.Server, 0)

	url, _ := url.Parse(api.RootUrl)
	host := url.Host
	basePath := api.BasePath
	if basePath == "" {
		basePath = "/"
	}
	d.Servers = append(d.Servers, &openapi3.Server{Url: "https://" + host + basePath})

	hasDataWrapper := false
	for _, feature := range api.Features {
		if feature == "dataWrapper" {
			hasDataWrapper = true
		}
	}

	d.Components = &openapi3.Components{}
	d.Components.Schemas = &openapi3.SchemasOrReferences{}
	if api.Schemas != nil {
		for _, pair := range api.Schemas.AdditionalProperties {
			addOpenAPI3SchemaForSchema(d, pair.Name, pair.Value)
		}
	}

	d.Paths = &openapi3.Paths{}
	if api.Methods != nil {
		for _, pair := range api.Methods.AdditionalProperties {
			addOpenAPI3PathsForMethod(d, pair.Name, pair.Value, hasDataWrapper)
		}
	}
	for _, pair := range api.Resources.AdditionalProperties {
		addOpenAPI3PathsForResource(d, pair.Value, hasDataWrapper)
	}

	return d, nil
}
