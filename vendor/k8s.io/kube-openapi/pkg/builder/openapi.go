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

package builder

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	restful "github.com/emicklei/go-restful"

	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/util"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const (
	OpenAPIVersion = "2.0"
)

type openAPI struct {
	config       *common.Config
	swagger      *spec.Swagger
	protocolList []string
	definitions  map[string]common.OpenAPIDefinition
}

// BuildOpenAPISpec builds OpenAPI spec given a list of route containers and common.Config to customize it.
//
// Deprecated: BuildOpenAPISpecFromRoutes should be used instead.
func BuildOpenAPISpec(routeContainers []*restful.WebService, config *common.Config) (*spec.Swagger, error) {
	return BuildOpenAPISpecFromRoutes(restfuladapter.AdaptWebServices(routeContainers), config)
}

// BuildOpenAPISpecFromRoutes builds OpenAPI spec given a list of route containers and common.Config to customize it.
func BuildOpenAPISpecFromRoutes(routeContainers []common.RouteContainer, config *common.Config) (*spec.Swagger, error) {
	o := newOpenAPI(config)
	err := o.buildPaths(routeContainers)
	if err != nil {
		return nil, err
	}
	return o.finalizeSwagger()
}

// BuildOpenAPIDefinitionsForResource builds a partial OpenAPI spec given a sample object and common.Config to customize it.
func BuildOpenAPIDefinitionsForResource(model interface{}, config *common.Config) (*spec.Definitions, error) {
	o := newOpenAPI(config)
	// We can discard the return value of toSchema because all we care about is the side effect of calling it.
	// All the models created for this resource get added to o.swagger.Definitions
	_, err := o.toSchema(util.GetCanonicalTypeName(model))
	if err != nil {
		return nil, err
	}
	swagger, err := o.finalizeSwagger()
	if err != nil {
		return nil, err
	}
	return &swagger.Definitions, nil
}

// BuildOpenAPIDefinitionsForResources returns the OpenAPI spec which includes the definitions for the
// passed type names.
func BuildOpenAPIDefinitionsForResources(config *common.Config, names ...string) (*spec.Swagger, error) {
	o := newOpenAPI(config)
	// We can discard the return value of toSchema because all we care about is the side effect of calling it.
	// All the models created for this resource get added to o.swagger.Definitions
	for _, name := range names {
		_, err := o.toSchema(name)
		if err != nil {
			return nil, err
		}
	}
	return o.finalizeSwagger()
}

// newOpenAPI sets up the openAPI object so we can build the spec.
func newOpenAPI(config *common.Config) openAPI {
	o := openAPI{
		config: config,
		swagger: &spec.Swagger{
			SwaggerProps: spec.SwaggerProps{
				Swagger:     OpenAPIVersion,
				Definitions: spec.Definitions{},
				Responses:   config.ResponseDefinitions,
				Paths:       &spec.Paths{Paths: map[string]spec.PathItem{}},
				Info:        config.Info,
			},
		},
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
	o.definitions = o.config.GetDefinitions(func(name string) spec.Ref {
		defName, _ := o.config.GetDefinitionName(name)
		return spec.MustCreateRef("#/definitions/" + common.EscapeJsonPointer(defName))
	})
	if o.config.CommonResponses == nil {
		o.config.CommonResponses = map[int]spec.Response{}
	}
	return o
}

// finalizeSwagger is called after the spec is built and returns the final spec.
// NOTE: finalizeSwagger also make changes to the final spec, as specified in the config.
func (o *openAPI) finalizeSwagger() (*spec.Swagger, error) {
	if o.config.SecurityDefinitions != nil {
		o.swagger.SecurityDefinitions = *o.config.SecurityDefinitions
		o.swagger.Security = o.config.DefaultSecurity
	}
	if o.config.PostProcessSpec != nil {
		var err error
		o.swagger, err = o.config.PostProcessSpec(o.swagger)
		if err != nil {
			return nil, err
		}
	}

	return o.swagger, nil
}

func (o *openAPI) buildDefinitionRecursively(name string) error {
	uniqueName, extensions := o.config.GetDefinitionName(name)
	if _, ok := o.swagger.Definitions[uniqueName]; ok {
		return nil
	}
	if item, ok := o.definitions[name]; ok {
		schema := spec.Schema{
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
		if v, ok := item.Schema.Extensions[common.ExtensionV2Schema]; ok {
			if v2Schema, isOpenAPISchema := v.(spec.Schema); isOpenAPISchema {
				schema = v2Schema
			}
		}
		o.swagger.Definitions[uniqueName] = schema
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

// buildDefinitionForType build a definition for a given type and return a referable name to its definition.
// This is the main function that keep track of definitions used in this spec and is depend on code generated
// by k8s.io/kubernetes/cmd/libs/go2idl/openapi-gen.
func (o *openAPI) buildDefinitionForType(name string) (string, error) {
	if err := o.buildDefinitionRecursively(name); err != nil {
		return "", err
	}
	defName, _ := o.config.GetDefinitionName(name)
	return "#/definitions/" + common.EscapeJsonPointer(defName), nil
}

// buildPaths builds OpenAPI paths using go-restful's web services.
func (o *openAPI) buildPaths(routeContainers []common.RouteContainer) error {
	pathsToIgnore := util.NewTrie(o.config.IgnorePrefixes)
	duplicateOpId := make(map[string]string)
	for _, w := range routeContainers {
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
			pathItem, exists := o.swagger.Paths.Paths[path]
			if exists {
				return fmt.Errorf("duplicate webservice route has been found for path: %v", path)
			}
			pathItem = spec.PathItem{
				PathItemProps: spec.PathItemProps{
					Parameters: make([]spec.Parameter, 0),
				},
			}
			// add web services's parameters as well as any parameters appears in all ops, as common parameters
			pathItem.Parameters = append(pathItem.Parameters, commonParams...)
			for _, p := range inPathCommonParamsMap {
				pathItem.Parameters = append(pathItem.Parameters, p)
			}
			sortParameters(pathItem.Parameters)
			for _, route := range routes {
				op, err := o.buildOperations(route, inPathCommonParamsMap)
				sortParameters(op.Parameters)
				if err != nil {
					return err
				}
				dpath, exists := duplicateOpId[op.ID]
				if exists {
					return fmt.Errorf("duplicate Operation ID %v for path %v and %v", op.ID, dpath, path)
				} else {
					duplicateOpId[op.ID] = path
				}
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
			o.swagger.Paths.Paths[path] = pathItem
		}
	}
	return nil
}

// buildOperations builds operations for each webservice path
func (o *openAPI) buildOperations(route common.Route, inPathCommonParamsMap map[interface{}]spec.Parameter) (ret *spec.Operation, err error) {
	ret = &spec.Operation{
		OperationProps: spec.OperationProps{
			Description: route.Description(),
			Consumes:    route.Consumes(),
			Produces:    route.Produces(),
			Schemes:     o.config.ProtocolList,
			Responses: &spec.Responses{
				ResponsesProps: spec.ResponsesProps{
					StatusCodeResponses: make(map[int]spec.Response),
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
	if ret.ID, ret.Tags, err = o.config.GetOperationIDAndTagsFromRoute(route); err != nil {
		return ret, err
	}

	// Build responses
	for _, resp := range route.StatusCodeResponses() {
		ret.Responses.StatusCodeResponses[resp.Code()], err = o.buildResponse(resp.Model(), resp.Message())
		if err != nil {
			return ret, err
		}
	}
	// If there is no response but a write sample, assume that write sample is an http.StatusOK response.
	if len(ret.Responses.StatusCodeResponses) == 0 && route.ResponsePayloadSample() != nil {
		ret.Responses.StatusCodeResponses[http.StatusOK], err = o.buildResponse(route.ResponsePayloadSample(), "OK")
		if err != nil {
			return ret, err
		}
	}
	for code, resp := range o.config.CommonResponses {
		if _, exists := ret.Responses.StatusCodeResponses[code]; !exists {
			ret.Responses.StatusCodeResponses[code] = resp
		}
	}
	// If there is still no response, use default response provided.
	if len(ret.Responses.StatusCodeResponses) == 0 {
		ret.Responses.Default = o.config.DefaultResponse
	}

	// Build non-common Parameters
	ret.Parameters = make([]spec.Parameter, 0)
	for _, param := range route.Parameters() {
		if _, isCommon := inPathCommonParamsMap[mapKeyFromParam(param)]; !isCommon {
			openAPIParam, err := o.buildParameter(param, route.RequestPayloadSample())
			if err != nil {
				return ret, err
			}
			ret.Parameters = append(ret.Parameters, openAPIParam)
		}
	}
	return ret, nil
}

func (o *openAPI) buildResponse(model interface{}, description string) (spec.Response, error) {
	schema, err := o.toSchema(util.GetCanonicalTypeName(model))
	if err != nil {
		return spec.Response{}, err
	}
	return spec.Response{
		ResponseProps: spec.ResponseProps{
			Description: description,
			Schema:      schema,
		},
	}, nil
}

func (o *openAPI) findCommonParameters(routes []common.Route) (map[interface{}]spec.Parameter, error) {
	commonParamsMap := make(map[interface{}]spec.Parameter, 0)
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
			openAPIParam, err := o.buildParameter(paramData, nil)
			if err != nil {
				return commonParamsMap, err
			}
			commonParamsMap[key] = openAPIParam
		}
	}
	return commonParamsMap, nil
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

func (o *openAPI) buildParameter(restParam common.Parameter, bodySample interface{}) (ret spec.Parameter, err error) {
	ret = spec.Parameter{
		ParamProps: spec.ParamProps{
			Name:        restParam.Name(),
			Description: restParam.Description(),
			Required:    restParam.Required(),
		},
	}
	switch restParam.Kind() {
	case common.BodyParameterKind:
		if bodySample != nil {
			ret.In = "body"
			ret.Schema, err = o.toSchema(util.GetCanonicalTypeName(bodySample))
			return ret, err
		} else {
			// There is not enough information in the body parameter to build the definition.
			// Body parameter has a data type that is a short name but we need full package name
			// of the type to create a definition.
			return ret, fmt.Errorf("restful body parameters are not supported: %v", restParam.DataType())
		}
	case common.PathParameterKind:
		ret.In = "path"
		if !restParam.Required() {
			return ret, fmt.Errorf("path parameters should be marked at required for parameter %v", restParam)
		}
	case common.QueryParameterKind:
		ret.In = "query"
	case common.HeaderParameterKind:
		ret.In = "header"
	case common.FormParameterKind:
		ret.In = "formData"
	default:
		return ret, fmt.Errorf("unknown restful operation kind : %v", restParam.Kind())
	}
	openAPIType, openAPIFormat := common.OpenAPITypeFormat(restParam.DataType())
	if openAPIType == "" {
		return ret, fmt.Errorf("non-body Restful parameter type should be a simple type, but got : %v", restParam.DataType())
	}
	ret.Type = openAPIType
	ret.Format = openAPIFormat
	ret.UniqueItems = !restParam.AllowMultiple()
	return ret, nil
}

func (o *openAPI) buildParameters(restParam []common.Parameter) (ret []spec.Parameter, err error) {
	ret = make([]spec.Parameter, len(restParam))
	for i, v := range restParam {
		ret[i], err = o.buildParameter(v, nil)
		if err != nil {
			return ret, err
		}
	}
	return ret, nil
}
