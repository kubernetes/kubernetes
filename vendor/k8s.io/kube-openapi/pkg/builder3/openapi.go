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
	"net/http"
	"strings"

	restful "github.com/emicklei/go-restful/v3"

	builderutil "k8s.io/kube-openapi/pkg/builder3/util"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/util"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const (
	OpenAPIVersion = "3.0"
)

type openAPI struct {
	config      *common.OpenAPIV3Config
	spec        *spec3.OpenAPI
	definitions map[string]common.OpenAPIDefinition
}

func groupRoutesByPath(routes []common.Route) map[string][]common.Route {
	pathToRoutes := make(map[string][]common.Route)
	for _, r := range routes {
		pathToRoutes[r.Path()] = append(pathToRoutes[r.Path()], r)
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

func (o *openAPI) buildOperations(route common.Route, inPathCommonParamsMap map[interface{}]*spec3.Parameter) (*spec3.Operation, error) {
	ret := &spec3.Operation{
		OperationProps: spec3.OperationProps{
			Description: route.Description(),
			Responses: &spec3.Responses{
				ResponsesProps: spec3.ResponsesProps{
					StatusCodeResponses: make(map[int]*spec3.Response),
				},
			},
		},
	}
	for k, v := range route.Metadata() {
		if strings.HasPrefix(k, common.ExtensionPrefix) {
			if ret.Extensions == nil {
				ret.Extensions = spec.Extensions{}
			}
			ret.Extensions.Add(k, v)
		}
	}

	var err error
	if ret.OperationId, ret.Tags, err = o.config.GetOperationIDAndTagsFromRoute(route); err != nil {
		return ret, err
	}

	// Build responses
	for _, resp := range route.StatusCodeResponses() {
		ret.Responses.StatusCodeResponses[resp.Code()], err = o.buildResponse(resp.Model(), resp.Message(), route.Produces())
		if err != nil {
			return ret, err
		}
	}

	// If there is no response but a write sample, assume that write sample is an http.StatusOK response.
	if len(ret.Responses.StatusCodeResponses) == 0 && route.ResponsePayloadSample() != nil {
		ret.Responses.StatusCodeResponses[http.StatusOK], err = o.buildResponse(route.ResponsePayloadSample(), "OK", route.Produces())
		if err != nil {
			return ret, err
		}
	}

	for code, resp := range o.config.CommonResponses {
		if _, exists := ret.Responses.StatusCodeResponses[code]; !exists {
			ret.Responses.StatusCodeResponses[code] = resp
		}
	}

	if len(ret.Responses.StatusCodeResponses) == 0 {
		ret.Responses.Default = o.config.DefaultResponse
	}

	params := route.Parameters()
	for _, param := range params {
		_, isCommon := inPathCommonParamsMap[mapKeyFromParam(param)]
		if !isCommon && param.Kind() != common.BodyParameterKind {
			openAPIParam, err := o.buildParameter(param)
			if err != nil {
				return ret, err
			}
			ret.Parameters = append(ret.Parameters, openAPIParam)
		}
	}

	body, err := o.buildRequestBody(params, route.Consumes(), route.RequestPayloadSample())
	if err != nil {
		return nil, err
	}

	if body != nil {
		ret.RequestBody = body
	}
	return ret, nil
}

func (o *openAPI) buildRequestBody(parameters []common.Parameter, consumes []string, bodySample interface{}) (*spec3.RequestBody, error) {
	for _, param := range parameters {
		if param.Kind() == common.BodyParameterKind && bodySample != nil {
			schema, err := o.toSchema(util.GetCanonicalTypeName(bodySample))
			if err != nil {
				return nil, err
			}
			r := &spec3.RequestBody{
				RequestBodyProps: spec3.RequestBodyProps{
					Content: map[string]*spec3.MediaType{},
				},
			}
			for _, consume := range consumes {
				r.Content[consume] = &spec3.MediaType{
					MediaTypeProps: spec3.MediaTypeProps{
						Schema: schema,
					},
				}
			}
			return r, nil
		}
	}
	return nil, nil
}

func newOpenAPI(config *common.Config) openAPI {
	o := openAPI{
		config: common.ConvertConfigToV3(config),
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
	if len(o.config.ResponseDefinitions) > 0 {
		o.spec.Components.Responses = make(map[string]*spec3.Response)

	}
	for k, response := range o.config.ResponseDefinitions {
		o.spec.Components.Responses[k] = response
	}

	if len(o.config.SecuritySchemes) > 0 {
		o.spec.Components.SecuritySchemes = make(spec3.SecuritySchemes)

	}
	for k, securityScheme := range o.config.SecuritySchemes {
		o.spec.Components.SecuritySchemes[k] = securityScheme
	}

	if o.config.GetOperationIDAndTagsFromRoute == nil {
		// Map the deprecated handler to the common interface, if provided.
		if o.config.GetOperationIDAndTags != nil {
			o.config.GetOperationIDAndTagsFromRoute = func(r common.Route) (string, []string, error) {
				restfulRouteAdapter, ok := r.(*restfuladapter.RouteAdapter)
				if !ok {
					return "", nil, fmt.Errorf("config.GetOperationIDAndTags specified but route is not a restful v1 Route")
				}

				return o.config.GetOperationIDAndTags(restfulRouteAdapter.Route)
			}
		} else {
			o.config.GetOperationIDAndTagsFromRoute = func(r common.Route) (string, []string, error) {
				return r.OperationName(), nil, nil
			}
		}
	}

	if o.config.GetDefinitionName == nil {
		o.config.GetDefinitionName = func(name string) (string, spec.Extensions) {
			return name[strings.LastIndex(name, "/")+1:], nil
		}
	}

	if o.config.Definitions != nil {
		o.definitions = o.config.Definitions
	} else {
		o.definitions = o.config.GetDefinitions(func(name string) spec.Ref {
			defName, _ := o.config.GetDefinitionName(name)
			return spec.MustCreateRef("#/components/schemas/" + common.EscapeJsonPointer(defName))
		})
	}

	return o
}

func (o *openAPI) buildOpenAPISpec(webServices []common.RouteContainer) error {
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
				PathProps: spec3.PathProps{},
			}

			// add web services's parameters as well as any parameters appears in all ops, as common parameters
			pathItem.Parameters = append(pathItem.Parameters, commonParams...)
			for _, p := range inPathCommonParamsMap {
				pathItem.Parameters = append(pathItem.Parameters, p)
			}
			sortParameters(pathItem.Parameters)

			for _, route := range routes {
				op, _ := o.buildOperations(route, inPathCommonParamsMap)
				sortParameters(op.Parameters)

				switch strings.ToUpper(route.Method()) {
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

// BuildOpenAPISpec builds OpenAPI v3 spec given a list of route containers and common.Config to customize it.
//
// Deprecated: BuildOpenAPISpecFromRoutes should be used instead.
func BuildOpenAPISpec(webServices []*restful.WebService, config *common.Config) (*spec3.OpenAPI, error) {
	return BuildOpenAPISpecFromRoutes(restfuladapter.AdaptWebServices(webServices), config)
}

// BuildOpenAPISpecFromRoutes builds OpenAPI v3 spec given a list of route containers and common.Config to customize it.
func BuildOpenAPISpecFromRoutes(webServices []common.RouteContainer, config *common.Config) (*spec3.OpenAPI, error) {
	a := newOpenAPI(config)
	err := a.buildOpenAPISpec(webServices)
	if err != nil {
		return nil, err
	}
	return a.spec, nil
}

func (o *openAPI) findCommonParameters(routes []common.Route) (map[interface{}]*spec3.Parameter, error) {
	commonParamsMap := make(map[interface{}]*spec3.Parameter, 0)
	paramOpsCountByName := make(map[interface{}]int, 0)
	paramNameKindToDataMap := make(map[interface{}]common.Parameter, 0)
	for _, route := range routes {
		routeParamDuplicateMap := make(map[interface{}]bool)
		s := ""
		params := route.Parameters()
		for _, param := range params {
			m, _ := json.Marshal(param)
			s += string(m) + "\n"
			key := mapKeyFromParam(param)
			if routeParamDuplicateMap[key] {
				msg, _ := json.Marshal(params)
				return commonParamsMap, fmt.Errorf("duplicate parameter %v for route %v, %v", param.Name(), string(msg), s)
			}
			routeParamDuplicateMap[key] = true
			paramOpsCountByName[key]++
			paramNameKindToDataMap[key] = param
		}
	}
	for key, count := range paramOpsCountByName {
		paramData := paramNameKindToDataMap[key]
		if count == len(routes) && paramData.Kind() != common.BodyParameterKind {
			openAPIParam, err := o.buildParameter(paramData)
			if err != nil {
				return commonParamsMap, err
			}
			commonParamsMap[key] = openAPIParam
		}
	}
	return commonParamsMap, nil
}

func (o *openAPI) buildParameters(restParam []common.Parameter) (ret []*spec3.Parameter, err error) {
	ret = make([]*spec3.Parameter, len(restParam))
	for i, v := range restParam {
		ret[i], err = o.buildParameter(v)
		if err != nil {
			return ret, err
		}
	}
	return ret, nil
}

func (o *openAPI) buildParameter(restParam common.Parameter) (ret *spec3.Parameter, err error) {
	ret = &spec3.Parameter{
		ParameterProps: spec3.ParameterProps{
			Name:        restParam.Name(),
			Description: restParam.Description(),
			Required:    restParam.Required(),
		},
	}
	switch restParam.Kind() {
	case common.BodyParameterKind:
		return nil, nil
	case common.PathParameterKind:
		ret.In = "path"
		if !restParam.Required() {
			return ret, fmt.Errorf("path parameters should be marked as required for parameter %v", restParam)
		}
	case common.QueryParameterKind:
		ret.In = "query"
	case common.HeaderParameterKind:
		ret.In = "header"
	/* TODO: add support for the cookie param */
	default:
		return ret, fmt.Errorf("unsupported restful parameter kind : %v", restParam.Kind())
	}
	openAPIType, openAPIFormat := common.OpenAPITypeFormat(restParam.DataType())
	if openAPIType == "" {
		return ret, fmt.Errorf("non-body Restful parameter type should be a simple type, but got : %v", restParam.DataType())
	}

	ret.Schema = &spec.Schema{
		SchemaProps: spec.SchemaProps{
			Type:        []string{openAPIType},
			Format:      openAPIFormat,
			UniqueItems: !restParam.AllowMultiple(),
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
		// delete the embedded v2 schema if exists, otherwise no-op
		delete(schema.VendorExtensible.Extensions, common.ExtensionV2Schema)
		schema = builderutil.WrapRefs(schema)
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
