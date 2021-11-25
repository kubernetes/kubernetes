/*
Copyright 2021 The Kubernetes Authors.

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

package builder3

import (
	"encoding/json"
	"fmt"
	restful "github.com/emicklei/go-restful"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/util"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"net/http"
	"strings"
)

const (
	OpenAPIVersion = "3.0"
)

type openAPI struct {
	config      *common.Config
	spec        *spec3.OpenAPI
	definitions map[string]common.OpenAPIDefinition
}

func groupRoutesByPath(routes []restful.Route) map[string][]restful.Route {
	pathToRoutes := make(map[string][]restful.Route)
	for _, r := range routes {
		pathToRoutes[r.Path] = append(pathToRoutes[r.Path], r)
	}
	return pathToRoutes
}

func (o *openAPI) buildResponse(model interface{}, description string, content []string) (*spec3.Response, error) {
	response := &spec3.Response{
		ResponseProps: spec3.ResponseProps{
			Description: description,
			Content:     make(map[string]*spec3.MediaType),
		},
	}

	s, err := o.toSchema(util.GetCanonicalTypeName(model))
	if err != nil {
		return nil, err
	}

	for _, contentType := range content {
		response.ResponseProps.Content[contentType] = &spec3.MediaType{
			MediaTypeProps: spec3.MediaTypeProps{
				Schema: s,
			},
		}
	}
	return response, nil
}

func (o *openAPI) buildOperations(route restful.Route, inPathCommonParamsMap map[interface{}]*spec3.Parameter) (*spec3.Operation, error) {
	ret := &spec3.Operation{
		OperationProps: spec3.OperationProps{
			Description: route.Doc,
			Responses: &spec3.Responses{
				ResponsesProps: spec3.ResponsesProps{
					StatusCodeResponses: make(map[int]*spec3.Response),
				},
			},
		},
	}
	var err error
	if ret.OperationId, ret.Tags, err = o.config.GetOperationIDAndTags(&route); err != nil {
		return ret, err
	}

	// Build responses
	for _, resp := range route.ResponseErrors {
		ret.Responses.StatusCodeResponses[resp.Code], err = o.buildResponse(resp.Model, resp.Message, route.Produces)
		if err != nil {
			return ret, err
		}
	}

	// If there is no response but a write sample, assume that write sample is an http.StatusOK response.
	if len(ret.Responses.StatusCodeResponses) == 0 && route.WriteSample != nil {
		ret.Responses.StatusCodeResponses[http.StatusOK], err = o.buildResponse(route.WriteSample, "OK", route.Produces)
		if err != nil {
			return ret, err
		}
	}

	// TODO: Default response if needed. Common Response config

	ret.Parameters = make([]*spec3.Parameter, 0)
	for _, param := range route.ParameterDocs {
		_, isCommon := inPathCommonParamsMap[mapKeyFromParam(param)]
		if !isCommon && param.Data().Kind != restful.BodyParameterKind {
			openAPIParam, err := o.buildParameter(param.Data())
			if err != nil {
				return ret, err
			}
			ret.Parameters = append(ret.Parameters, openAPIParam)
		}
	}

	body, err := o.buildRequestBody(route.ParameterDocs, route.ReadSample)
	if err != nil {
		return nil ,err
	}

	if body != nil {
		ret.RequestBody = body
	}
	return ret, nil
}

func (o *openAPI) buildRequestBody(parameters []*restful.Parameter, bodySample interface{}) (*spec3.RequestBody, error) {
	for _, param := range parameters {
		if param.Data().Kind == restful.BodyParameterKind && bodySample != nil {
			schema, err := o.toSchema(util.GetCanonicalTypeName(bodySample))
			if err != nil {
				return nil, err
			}
			r := &spec3.RequestBody{
				RequestBodyProps: spec3.RequestBodyProps{
					Content: map[string]*spec3.MediaType{
						"application/json": &spec3.MediaType{
							MediaTypeProps: spec3.MediaTypeProps{
								Schema: schema,
							},
						},
					},
				},
			}
			return r, nil
		}
	}
	return nil, nil
}

func newOpenAPI(config *common.Config) openAPI {
	o := openAPI{
		config: config,
		spec: &spec3.OpenAPI{
			Version: "3.0.0",
			Info:    config.Info,
			Paths: &spec3.Paths{
				Paths: map[string]*spec3.Path{},
			},
			Components: &spec3.Components{
				Schemas: map[string]*spec.Schema{},
			},
		},
	}
	if o.config.GetOperationIDAndTags == nil {
		o.config.GetOperationIDAndTags = func(r *restful.Route) (string, []string, error) {
			return r.Operation, nil, nil
		}
	}

	if o.config.GetDefinitionName == nil {
		o.config.GetDefinitionName = func(name string) (string, spec.Extensions) {
			return name[strings.LastIndex(name, "/")+1:], nil
		}
	}

	o.definitions = o.config.GetDefinitions(func(name string) spec.Ref {
		defName, _ := o.config.GetDefinitionName(name)
		return spec.MustCreateRef("#/components/schemas/" + common.EscapeJsonPointer(defName))
	})

	return o
}

func (o *openAPI) buildOpenAPISpec(webServices []*restful.WebService) error {
	pathsToIgnore := util.NewTrie(o.config.IgnorePrefixes)
	for _, w := range webServices {
		rootPath := w.RootPath()
		if pathsToIgnore.HasPrefix(rootPath) {
			continue
		}

		commonParams, err := o.buildParameters(w.PathParameters())
		if err != nil {
			return err
		}

		for path, routes := range groupRoutesByPath(w.Routes()) {
			// go-swagger has special variable definition {$NAME:*} that can only be
			// used at the end of the path and it is not recognized by OpenAPI.
			if strings.HasSuffix(path, ":*}") {
				path = path[:len(path)-3] + "}"
			}
			if pathsToIgnore.HasPrefix(path) {
				continue
			}

			// Aggregating common parameters make API spec (and generated clients) simpler
			inPathCommonParamsMap, err := o.findCommonParameters(routes)
			if err != nil {
				return err
			}
			pathItem, exists := o.spec.Paths.Paths[path]
			if exists {
				return fmt.Errorf("duplicate webservice route has been found for path: %v", path)
			}

			pathItem = &spec3.Path{
				PathProps: spec3.PathProps{
					Parameters: make([]*spec3.Parameter, 0),
				},
			}

			// add web services's parameters as well as any parameters appears in all ops, as common parameters
			pathItem.Parameters = append(pathItem.Parameters, commonParams...)
			for _, p := range inPathCommonParamsMap {
				pathItem.Parameters = append(pathItem.Parameters, p)
			}
			sortParameters(pathItem.Parameters)

			for _, route := range routes {
				op, _ := o.buildOperations(route, inPathCommonParamsMap)

				switch strings.ToUpper(route.Method) {
				case "GET":
					pathItem.Get = op
				case "POST":
					pathItem.Post = op
				case "HEAD":
					pathItem.Head = op
				case "PUT":
					pathItem.Put = op
				case "DELETE":
					pathItem.Delete = op
				case "OPTIONS":
					pathItem.Options = op
				case "PATCH":
					pathItem.Patch = op
				}

			}
			o.spec.Paths.Paths[path] = pathItem
		}
	}
	return nil
}

func BuildOpenAPISpec(webServices []*restful.WebService, config *common.Config) (*spec3.OpenAPI, error) {
	a := newOpenAPI(config)
	err := a.buildOpenAPISpec(webServices)
	if err != nil {
		return nil, err
	}
	return a.spec, nil
}

func (o *openAPI) findCommonParameters(routes []restful.Route) (map[interface{}]*spec3.Parameter, error) {
	commonParamsMap := make(map[interface{}]*spec3.Parameter, 0)
	paramOpsCountByName := make(map[interface{}]int, 0)
	paramNameKindToDataMap := make(map[interface{}]restful.ParameterData, 0)
	for _, route := range routes {
		routeParamDuplicateMap := make(map[interface{}]bool)
		s := ""
		for _, param := range route.ParameterDocs {
			m, _ := json.Marshal(param.Data())
			s += string(m) + "\n"
			key := mapKeyFromParam(param)
			if routeParamDuplicateMap[key] {
				msg, _ := json.Marshal(route.ParameterDocs)
				return commonParamsMap, fmt.Errorf("duplicate parameter %v for route %v, %v", param.Data().Name, string(msg), s)
			}
			routeParamDuplicateMap[key] = true
			paramOpsCountByName[key]++
			paramNameKindToDataMap[key] = param.Data()
		}
	}
	for key, count := range paramOpsCountByName {
		paramData := paramNameKindToDataMap[key]
		if count == len(routes) && paramData.Kind != restful.BodyParameterKind {
			openAPIParam, err := o.buildParameter(paramData)
			if err != nil {
				return commonParamsMap, err
			}
			commonParamsMap[key] = openAPIParam
		}
	}
	return commonParamsMap, nil
}

func (o *openAPI) buildParameters(restParam []*restful.Parameter) (ret []*spec3.Parameter, err error) {
	ret = make([]*spec3.Parameter, len(restParam))
	for i, v := range restParam {
		ret[i], err = o.buildParameter(v.Data())
		if err != nil {
			return ret, err
		}
	}
	return ret, nil
}

func (o *openAPI) buildParameter(restParam restful.ParameterData) (ret *spec3.Parameter, err error) {
	ret = &spec3.Parameter{
		ParameterProps: spec3.ParameterProps{
			Name:        restParam.Name,
			Description: restParam.Description,
			Required:    restParam.Required,
		},
	}
	switch restParam.Kind {
	case restful.BodyParameterKind:
		return nil, nil
	case restful.PathParameterKind:
		ret.In = "path"
		if !restParam.Required {
			return ret, fmt.Errorf("path parameters should be marked at required for parameter %v", restParam)
		}
	case restful.QueryParameterKind:
		ret.In = "query"
	case restful.HeaderParameterKind:
		ret.In = "header"
	/* TODO: add support for the cookie param */
	default:
		return ret, fmt.Errorf("unsupported restful parameter kind : %v", restParam.Kind)
	}
	openAPIType, openAPIFormat := common.OpenAPITypeFormat(restParam.DataType)
	if openAPIType == "" {
		return ret, fmt.Errorf("non-body Restful parameter type should be a simple type, but got : %v", restParam.DataType)
	}

	ret.Schema = &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Type:        []string{openAPIType},
			Format:      openAPIFormat,
			UniqueItems: !restParam.AllowMultiple,
		},
	}
	return ret, nil
}

func (o *openAPI) buildDefinitionRecursively(name string) error {
	uniqueName, extensions := o.config.GetDefinitionName(name)
	if _, ok := o.spec.Components.Schemas[uniqueName]; ok {
		return nil
	}
	if item, ok := o.definitions[name]; ok {
		schema := &spec.Schema{
			VendorExtensible:   item.Schema.VendorExtensible,
			SchemaProps:        item.Schema.SchemaProps,
			SwaggerSchemaProps: item.Schema.SwaggerSchemaProps,
		}
		if extensions != nil {
			if schema.Extensions == nil {
				schema.Extensions = spec.Extensions{}
			}
			for k, v := range extensions {
				schema.Extensions[k] = v
			}
		}
		o.spec.Components.Schemas[uniqueName] = schema
		for _, v := range item.Dependencies {
			if err := o.buildDefinitionRecursively(v); err != nil {
				return err
			}
		}
	} else {
		return fmt.Errorf("cannot find model definition for %v. If you added a new type, you may need to add +k8s:openapi-gen=true to the package or type and run code-gen again", name)
	}
	return nil
}

func (o *openAPI) buildDefinitionForType(name string) (string, error) {
	if err := o.buildDefinitionRecursively(name); err != nil {
		return "", err
	}
	defName, _ := o.config.GetDefinitionName(name)
	return "#/components/schemas/" + common.EscapeJsonPointer(defName), nil
}

func (o *openAPI) toSchema(name string) (_ *spec.Schema, err error) {
	if openAPIType, openAPIFormat := common.OpenAPITypeFormat(name); openAPIType != "" {
		return &spec.Schema{
			SchemaProps: spec.SchemaProps{
				Type:   []string{openAPIType},
				Format: openAPIFormat,
			},
		}, nil
	} else {
		ref, err := o.buildDefinitionForType(name)
		if err != nil {
			return nil, err
		}
		return &spec.Schema{
			SchemaProps: spec.SchemaProps{
				Ref: spec.MustCreateRef(ref),
			},
		}, nil
	}
}
