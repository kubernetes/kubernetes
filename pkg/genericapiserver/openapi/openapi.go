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
	"reflect"
	"strings"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	"k8s.io/kubernetes/pkg/util/json"
)

const (
	OpenAPIVersion = "2.0"
)

// Config is set of configuration for openAPI spec generation.
type Config struct {
	// Path to the spec file. by convention, it should name [.*/]*/swagger.json
	OpenAPIServePath string
	// List of web services for this API spec
	WebServices []*restful.WebService

	// List of supported protocols such as https, http, etc.
	ProtocolList []string

	// Info is general information about the API.
	Info *spec.Info
	// DefaultResponse will be used if an operation does not have any responses listed. It
	// will show up as ... "responses" : {"default" : $DefaultResponse} in the spec.
	DefaultResponse *spec.Response
	// List of webservice's path prefixes to ignore
	IgnorePrefixes []string
}

type openAPI struct {
	config       *Config
	swagger      *spec.Swagger
	definitions  spec.Definitions
	protocolList []string
}

type OpenAPIType struct {
	Schema       *spec.Schema
	Dependencies map[string]interface{}
}

type OpenAPIGetType interface {
	OpenAPI() OpenAPIType
}

// RegisterOpenAPIService registers a handler to provides standard OpenAPI specification.
func RegisterOpenAPIService(config *Config, containers *restful.Container) (err error) {
	o := openAPI{
		config:      config,
		definitions: spec.Definitions{},
	}
	err = o.buildSpec()
	if err != nil {
		return err
	}
	containers.ServeMux.HandleFunc(config.OpenAPIServePath, func(w http.ResponseWriter, r *http.Request) {
		resp := restful.NewResponse(w)
		if r.URL.Path != config.OpenAPIServePath {
			resp.WriteErrorString(http.StatusNotFound, "Path not found!")
		}
		resp.WriteAsJson(o.swagger)
	})
	return nil
}

func (o *openAPI) buildSpec() (err error) {
	if o.swagger != nil {
		return fmt.Errorf("OpenAPI spec is already built. Duplicate call to buildSpec is not allowed.")
	}
	paths, err := o.buildPaths()
	if err != nil {
		return err
	}
	o.swagger = &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Swagger:     OpenAPIVersion,
			Definitions: o.definitions,
			Paths:       &paths,
			Info:        o.config.Info,
		},
	}
	return nil
}

// buildDefinitionForType build a definition for a given type and return a referable name to it's definition.
// This is the main function that keep track of definitions used in this spec and is depend on code generated
// by k8s.io/kubernetes/cmd/libs/go2idl/openapi-gen.
func (o *openAPI) buildDefinitionForType(sample interface{}) (string, error) {
	t := reflect.TypeOf(sample)
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}
	name := t.String()
	_, ok := o.definitions[name]
	if !ok {
		if openAPIGetType, ok := sample.(OpenAPIGetType); ok {
			i := openAPIGetType.OpenAPI()
			o.definitions[name] = *i.Schema
			for _, v := range i.Dependencies {
				if _, err := o.buildDefinitionForType(v); err != nil {
					return "", err
				}
			}
		} else {
			return "", fmt.Errorf("Cannot find model definition for %v.", t)
		}
	}
	return "#/definitions/" + name, nil
}

// buildPaths builds OpenAPI paths using go-restful's web services.
func (o *openAPI) buildPaths() (spec.Paths, error) {
	paths := spec.Paths{
		Paths: make(map[string]spec.PathItem),
	}
	pathsToIgnore := createTrie(o.config.IgnorePrefixes)
	duplicateOpId := make(map[string]bool)
	// Find duplicate operation IDs.
	for _, service := range o.config.WebServices {
		if pathsToIgnore.HasPrefix(service.RootPath()) {
			continue
		}
		for _, route := range service.Routes() {
			_, exists := duplicateOpId[route.Operation]
			duplicateOpId[route.Operation] = exists
		}
	}
	for _, w := range o.config.WebServices {
		rootPath := w.RootPath()
		if pathsToIgnore.HasPrefix(rootPath) {
			continue
		}
		commonParams, err := o.buildParameters(w.PathParameters())
		if err != nil {
			return paths, err
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
				return paths, err
			}
			pathItem, exists := paths.Paths[path]
			if exists {
				return paths, fmt.Errorf("Duplicate webservice route has been found for path: %v", path)
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
			for _, route := range routes {
				op, err := o.buildOperations(route, inPathCommonParamsMap)
				if err != nil {
					return paths, err
				}
				if duplicateOpId[op.ID] {
					// Repeated Operation IDs are not allowed in OpenAPI spec but if
					// an OperationID is empty, client generators will infer the ID
					// from the path and method of operation.
					op.ID = ""
				}
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
			paths.Paths[path] = pathItem
		}
	}
	return paths, nil
}

// buildOperations builds operations for each webservice path
func (o *openAPI) buildOperations(route restful.Route, inPathCommonParamsMap map[interface{}]spec.Parameter) (ret *spec.Operation, err error) {
	ret = &spec.Operation{
		OperationProps: spec.OperationProps{
			Description: route.Doc,
			Consumes:    route.Consumes,
			Produces:    route.Produces,
			ID:          route.Operation,
			Schemes:     o.config.ProtocolList,
			Responses: &spec.Responses{
				ResponsesProps: spec.ResponsesProps{
					StatusCodeResponses: make(map[int]spec.Response),
				},
			},
		},
	}

	// Build responses
	for _, resp := range route.ResponseErrors {
		ret.Responses.StatusCodeResponses[resp.Code], err = o.buildResponse(resp.Model, resp.Message)
		if err != nil {
			return ret, err
		}
	}
	// If there is no response but a write sample, assume that write sample is an http.StatusOK response.
	if len(ret.Responses.StatusCodeResponses) == 0 && route.WriteSample != nil {
		ret.Responses.StatusCodeResponses[http.StatusOK], err = o.buildResponse(route.WriteSample, "OK")
		if err != nil {
			return ret, err
		}
	}
	// If there is still no response, use default response provided.
	if len(ret.Responses.StatusCodeResponses) == 0 {
		ret.Responses.Default = o.config.DefaultResponse
	}
	// If there is a read sample, there will be a body param referring to it.
	if route.ReadSample != nil {
		if _, err := o.toSchema(reflect.TypeOf(route.ReadSample).String(), route.ReadSample); err != nil {
			return ret, err
		}
	}

	// Build non-common Parameters
	ret.Parameters = make([]spec.Parameter, 0)
	for _, param := range route.ParameterDocs {
		if _, isCommon := inPathCommonParamsMap[mapKeyFromParam(param)]; !isCommon {
			openAPIParam, err := o.buildParameter(param.Data())
			if err != nil {
				return ret, err
			}
			ret.Parameters = append(ret.Parameters, openAPIParam)
		}
	}
	return ret, nil
}

func (o *openAPI) buildResponse(model interface{}, description string) (spec.Response, error) {
	typeName := reflect.TypeOf(model).String()
	schema, err := o.toSchema(typeName, model)
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

func groupRoutesByPath(routes []restful.Route) (ret map[string][]restful.Route) {
	ret = make(map[string][]restful.Route)
	for _, r := range routes {
		route, exists := ret[r.Path]
		if !exists {
			route = make([]restful.Route, 0, 1)
		}
		ret[r.Path] = append(route, r)
	}
	return ret
}

func mapKeyFromParam(param *restful.Parameter) interface{} {
	return struct {
		Name string
		Kind int
	}{
		Name: param.Data().Name,
		Kind: param.Data().Kind,
	}
}

func (o *openAPI) findCommonParameters(routes []restful.Route) (map[interface{}]spec.Parameter, error) {
	commonParamsMap := make(map[interface{}]spec.Parameter, 0)
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
				return commonParamsMap, fmt.Errorf("Duplicate parameter %v for route %v, %v.", param.Data().Name, string(msg), s)
			}
			routeParamDuplicateMap[key] = true
			paramOpsCountByName[key]++
			paramNameKindToDataMap[key] = param.Data()
		}
	}
	for key, count := range paramOpsCountByName {
		if count == len(routes) {
			openAPIParam, err := o.buildParameter(paramNameKindToDataMap[key])
			if err != nil {
				return commonParamsMap, err
			}
			commonParamsMap[key] = openAPIParam
		}
	}
	return commonParamsMap, nil
}

func (o *openAPI) toSchema(typeName string, model interface{}) (_ *spec.Schema, err error) {
	if openAPIType, openAPIFormat := GetOpenAPITypeFormat(typeName); openAPIType != "" {
		return &spec.Schema{
			SchemaProps: spec.SchemaProps{
				Type:   []string{openAPIType},
				Format: openAPIFormat,
			},
		}, nil
	} else {
		ref := "#/definitions/" + typeName
		if model != nil {
			ref, err = o.buildDefinitionForType(model)
			if err != nil {
				return nil, err
			}
		}
		return &spec.Schema{
			SchemaProps: spec.SchemaProps{
				Ref: spec.MustCreateRef(ref),
			},
		}, nil
	}
}

func (o *openAPI) buildParameter(restParam restful.ParameterData) (ret spec.Parameter, err error) {
	ret = spec.Parameter{
		ParamProps: spec.ParamProps{
			Name:        restParam.Name,
			Description: restParam.Description,
			Required:    restParam.Required,
		},
	}
	switch restParam.Kind {
	case restful.BodyParameterKind:
		ret.In = "body"
		ret.Schema, err = o.toSchema(restParam.DataType, nil)
		return ret, err
	case restful.PathParameterKind:
		ret.In = "path"
		if !restParam.Required {
			return ret, fmt.Errorf("Path parameters should be marked at required for parameter %v", restParam)
		}
	case restful.QueryParameterKind:
		ret.In = "query"
	case restful.HeaderParameterKind:
		ret.In = "header"
	case restful.FormParameterKind:
		ret.In = "form"
	default:
		return ret, fmt.Errorf("Unknown restful operation kind : %v", restParam.Kind)
	}
	openAPIType, openAPIFormat := GetOpenAPITypeFormat(restParam.DataType)
	if openAPIType == "" {
		return ret, fmt.Errorf("Non-Body Restful parameter type should be a simple type, but got : %v", restParam.DataType)
	}
	ret.Type = openAPIType
	ret.Format = openAPIFormat
	ret.UniqueItems = !restParam.AllowMultiple
	return ret, nil
}

func (o *openAPI) buildParameters(restParam []*restful.Parameter) (ret []spec.Parameter, err error) {
	ret = make([]spec.Parameter, len(restParam))
	for i, v := range restParam {
		ret[i], err = o.buildParameter(v.Data())
		if err != nil {
			return ret, err
		}
	}
	return ret, nil
}

// This function is a reference for converting go (or any custom type) to a simple open API type,format pair
// Is being used in spec generation as well as code generation.
func GetOpenAPITypeFormat(typeName string) (string, string) {
	schemaTypeFormatMap := map[string][]string{
		"uint":      {"integer", "int32"},
		"uint8":     {"integer", "byte"},
		"uint16":    {"integer", "int32"},
		"uint32":    {"integer", "int64"},
		"uint64":    {"integer", "int64"},
		"int":       {"integer", "int32"},
		"int8":      {"integer", "byte"},
		"int16":     {"integer", "int32"},
		"int32":     {"integer", "int32"},
		"int64":     {"integer", "int64"},
		"byte":      {"integer", "byte"},
		"float64":   {"number", "double"},
		"float32":   {"number", "float"},
		"bool":      {"boolean", ""},
		"time.Time": {"string", "date-time"},
		"string":    {"string", ""},

		// base64 encoded characters
		"[]byte": {"string", "byte"},

		"k8s.io/kubernetes/pkg/runtime.Object":       {"string", ""},
		"k8s.io/kubernetes/pkg/api/unversioned.Time": {"string", "date-time"},

		// First step of supporting int-or-string. The format is a free field in open api.
		"k8s.io/kubernetes/pkg/util/intstr.IntOrString": {"string", "int-or-string"},

		// We do not use UID only for UUID thus we assume it is string
		"k8s.io/kubernetes/pkg/types.UID": {"string", ""},

		"k8s.io/kubernetes/pkg/api.Protocol":    {"string", ""},
		"k8s.io/kubernetes/pkg/api/v1.Protocol": {"string", ""},

		// This type has complex dependencies and we are going to ignore it for now to
		// make the spec valid.
		// TODO: Fix this type and API's depending on it.
		"k8s.io/kubernetes/pkg/api/resource.infDecAmount": {"string", ""},

		"integer": {"integer", ""},
		"number":  {"number", ""},
		"boolean": {"boolean", ""},
	}
	mapped, ok := schemaTypeFormatMap[typeName]
	if !ok {
		return "", ""
	}
	return mapped[0], mapped[1]
}

// A simple trie implementation with Add an HasPrefix methods only.
type trie struct {
	children map[byte]*trie
	wordTail bool
}

func createTrie(list []string) trie {
	ret := trie{
		children: make(map[byte]*trie),
		wordTail: false,
	}
	for _, v := range list {
		ret.Add(v)
	}
	return ret
}

func (t *trie) Add(v string) {
	root := t
	for _, b := range []byte(v) {
		child, exists := root.children[b]
		if !exists {
			child = &trie{
				children: make(map[byte]*trie),
				wordTail: false,
			}
			root.children[b] = child
		}
		root = child
	}
	root.wordTail = true
}

func (t *trie) HasPrefix(v string) bool {
	root := t
	if root.wordTail {
		return true
	}
	for _, b := range []byte(v) {
		child, exists := root.children[b]
		if !exists {
			return false
		}
		if child.wordTail {
			return true
		}
		root = child
	}
	return false
}
