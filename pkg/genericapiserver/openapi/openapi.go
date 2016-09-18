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

// Note: Any reference to swagger in this document is to swagger 1.2 spec.

import (
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"sort"
	"strconv"
	"strings"

	"github.com/emicklei/go-restful"
	"github.com/emicklei/go-restful/swagger"
	"github.com/go-openapi/loads"
	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/validate"
	"k8s.io/kubernetes/pkg/util/json"
)

const (
	// By convention, the Swagger specification file is named swagger.json
	OpenAPIServePath = "/swagger.json"
	OpenAPIVersion   = "2.0"
)

// Config is set of configuration for openAPI spec generation.
type Config struct {
	// SwaggerConfig is set of configuration for go-restful swagger spec generation. Currently
	// openAPI implementation depends on go-restful to generate models.
	SwaggerConfig *swagger.Config
	// Info is general information about the API.
	Info *spec.Info
	// DefaultResponse will be used if an operation does not have any responses listed. It
	// will show up as ... "responses" : {"default" : $DefaultResponse} in swagger spec.
	DefaultResponse *spec.Response
	// List of webservice's path prefixes to ignore
	IgnorePrefixes []string
}

type openAPI struct {
	config       *Config
	swagger      *spec.Swagger
	protocolList []string
}

// RegisterOpenAPIService registers a handler to provides standard OpenAPI specification.
func RegisterOpenAPIService(config *Config, containers *restful.Container) (err error) {
	var _ = loads.Spec
	var _ = strfmt.ParseDuration
	var _ = validate.FormatOf
	o := openAPI{
		config: config,
	}
	err = o.buildSwaggerSpec()
	if err != nil {
		return err
	}
	containers.ServeMux.HandleFunc(OpenAPIServePath, func(w http.ResponseWriter, r *http.Request) {
		resp := restful.NewResponse(w)
		if r.URL.Path != OpenAPIServePath {
			resp.WriteErrorString(http.StatusNotFound, "Path not found!")
		}
		resp.WriteAsJson(o.swagger)
	})
	return nil
}

func (o *openAPI) buildSwaggerSpec() (err error) {
	if o.swagger != nil {
		return fmt.Errorf("Swagger spec is already built. Duplicate call to buildSwaggerSpec is not allowed.")
	}
	o.protocolList, err = o.buildProtocolList()
	if err != nil {
		return err
	}
	definitions, err := o.buildDefinitions()
	if err != nil {
		return err
	}
	paths, err := o.buildPaths()
	if err != nil {
		return err
	}
	o.swagger = &spec.Swagger{
		SwaggerProps: spec.SwaggerProps{
			Swagger:     OpenAPIVersion,
			Definitions: definitions,
			Paths:       &paths,
			Info:        o.config.Info,
		},
	}
	return nil
}

// buildDefinitions construct OpenAPI definitions using go-restful's swagger 1.2 generated models.
func (o *openAPI) buildDefinitions() (definitions spec.Definitions, err error) {
	definitions = spec.Definitions{}
	for _, decl := range swagger.NewSwaggerBuilder(*o.config.SwaggerConfig).ProduceAllDeclarations() {
		for _, swaggerModel := range decl.Models.List {
			_, ok := definitions[swaggerModel.Name]
			if ok {
				// TODO(mbohlool): decide what to do with repeated models
				// The best way is to make sure they have the same content and
				// fail otherwise.
				continue
			}
			definitions[swaggerModel.Name], err = buildModel(swaggerModel.Model)
			if err != nil {
				return definitions, err
			}
		}
	}
	return definitions, nil
}

func buildModel(swaggerModel swagger.Model) (ret spec.Schema, err error) {
	ret = spec.Schema{
		// SchemaProps.SubTypes is not used in go-restful, ignoring.
		SchemaProps: spec.SchemaProps{
			Description: swaggerModel.Description,
			Required:    swaggerModel.Required,
			Properties:  make(map[string]spec.Schema),
		},
		SwaggerSchemaProps: spec.SwaggerSchemaProps{
			Discriminator: swaggerModel.Discriminator,
		},
	}
	for _, swaggerProp := range swaggerModel.Properties.List {
		if _, ok := ret.Properties[swaggerProp.Name]; ok {
			return ret, fmt.Errorf("Duplicate property in swagger 1.2 spec: %v", swaggerProp.Name)
		}
		ret.Properties[swaggerProp.Name], err = buildProperty(swaggerProp)
		if err != nil {
			return ret, err
		}
	}
	return ret, nil
}

// buildProperty converts a swagger 1.2 property to an open API property.
func buildProperty(swaggerProperty swagger.NamedModelProperty) (openAPIProperty spec.Schema, err error) {
	if swaggerProperty.Property.Ref != nil {
		return spec.Schema{
			SchemaProps: spec.SchemaProps{
				Ref: spec.MustCreateRef("#/definitions/" + *swaggerProperty.Property.Ref),
			},
		}, nil
	}
	openAPIProperty = spec.Schema{
		SchemaProps: spec.SchemaProps{
			Description: swaggerProperty.Property.Description,
			Default:     getDefaultValue(swaggerProperty.Property.DefaultValue),
			Enum:        make([]interface{}, len(swaggerProperty.Property.Enum)),
		},
	}
	for i, e := range swaggerProperty.Property.Enum {
		openAPIProperty.Enum[i] = e
	}
	openAPIProperty.Minimum, err = getFloat64OrNil(swaggerProperty.Property.Minimum)
	if err != nil {
		return spec.Schema{}, err
	}
	openAPIProperty.Maximum, err = getFloat64OrNil(swaggerProperty.Property.Maximum)
	if err != nil {
		return spec.Schema{}, err
	}
	if swaggerProperty.Property.UniqueItems != nil {
		openAPIProperty.UniqueItems = *swaggerProperty.Property.UniqueItems
	}

	if swaggerProperty.Property.Items != nil {
		if swaggerProperty.Property.Items.Ref != nil {
			openAPIProperty.Items = &spec.SchemaOrArray{
				Schema: &spec.Schema{
					SchemaProps: spec.SchemaProps{
						Ref: spec.MustCreateRef("#/definitions/" + *swaggerProperty.Property.Items.Ref),
					},
				},
			}
		} else {
			openAPIProperty.Items = &spec.SchemaOrArray{
				Schema: &spec.Schema{},
			}
			openAPIProperty.Items.Schema.Type, openAPIProperty.Items.Schema.Format, err =
				buildType(swaggerProperty.Property.Items.Type, swaggerProperty.Property.Items.Format)
			if err != nil {
				return spec.Schema{}, err
			}
		}
	}
	openAPIProperty.Type, openAPIProperty.Format, err =
		buildType(swaggerProperty.Property.Type, swaggerProperty.Property.Format)
	if err != nil {
		return spec.Schema{}, err
	}
	return openAPIProperty, nil
}

// buildPaths builds OpenAPI paths using go-restful's web services.
func (o *openAPI) buildPaths() (spec.Paths, error) {
	paths := spec.Paths{
		Paths: make(map[string]spec.PathItem),
	}
	pathsToIgnore := createTrie(o.config.IgnorePrefixes)
	duplicateOpId := make(map[string]bool)
	// Find duplicate operation IDs.
	for _, service := range o.config.SwaggerConfig.WebServices {
		if pathsToIgnore.HasPrefix(service.RootPath()) {
			continue
		}
		for _, route := range service.Routes() {
			_, exists := duplicateOpId[route.Operation]
			duplicateOpId[route.Operation] = exists
		}
	}
	for _, w := range o.config.SwaggerConfig.WebServices {
		rootPath := w.RootPath()
		if pathsToIgnore.HasPrefix(rootPath) {
			continue
		}
		commonParams, err := buildParameters(w.PathParameters())
		if err != nil {
			return paths, err
		}
		sort.Sort(orderedParameters(commonParams))
		for path, routes := range groupRoutesByPath(w.Routes()) {
			// go-swagger has special variable difinition {$NAME:*} that can only be
			// used at the end of the path and it is not recognized by OpenAPI.
			if strings.HasSuffix(path, ":*}") {
				path = path[:len(path)-3] + "}"
			}
			inPathCommonParamsMap, err := findCommonParameters(routes)
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
			for _, p := range commonParams {
				pathItem.Parameters = append(pathItem.Parameters, p)
			}
			for _, p := range inPathCommonParamsMap {
				pathItem.Parameters = append(pathItem.Parameters, p)
			}
			sort.Sort(orderedParameters(pathItem.Parameters))
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

// buildProtocolList returns list of accepted protocols for this web service. If web service url has no protocol, it
// will default to http.
func (o *openAPI) buildProtocolList() ([]string, error) {
	uri, err := url.Parse(o.config.SwaggerConfig.WebServicesUrl)
	if err != nil {
		return []string{}, err
	}
	if uri.Scheme != "" {
		return []string{uri.Scheme}, nil
	} else {
		return []string{"http"}, nil
	}
}

// buildOperations builds operations for each webservice path
func (o *openAPI) buildOperations(route restful.Route, inPathCommonParamsMap map[interface{}]spec.Parameter) (*spec.Operation, error) {
	ret := &spec.Operation{
		OperationProps: spec.OperationProps{
			Description: route.Doc,
			Consumes:    route.Consumes,
			Produces:    route.Produces,
			ID:          route.Operation,
			Schemes:     o.protocolList,
			Responses: &spec.Responses{
				ResponsesProps: spec.ResponsesProps{
					StatusCodeResponses: make(map[int]spec.Response),
				},
			},
		},
	}
	for _, resp := range route.ResponseErrors {
		ret.Responses.StatusCodeResponses[resp.Code] = spec.Response{
			ResponseProps: spec.ResponseProps{
				Description: resp.Message,
				Schema: &spec.Schema{
					SchemaProps: spec.SchemaProps{
						Ref: spec.MustCreateRef("#/definitions/" + reflect.TypeOf(resp.Model).String()),
					},
				},
			},
		}
	}
	if len(ret.Responses.StatusCodeResponses) == 0 {
		ret.Responses.Default = o.config.DefaultResponse
	}
	ret.Parameters = make([]spec.Parameter, 0)
	for _, param := range route.ParameterDocs {
		_, isCommon := inPathCommonParamsMap[mapKeyFromParam(param)]
		if !isCommon {
			openAPIParam, err := buildParameter(param.Data())
			if err != nil {
				return ret, err
			}
			ret.Parameters = append(ret.Parameters, openAPIParam)
		}
	}
	sort.Sort(orderedParameters(ret.Parameters))
	return ret, nil
}

type orderedParameters []spec.Parameter

func (p orderedParameters) Len() int      { return len(p) }
func (p orderedParameters) Swap(i, j int) { p[i], p[j] = p[j], p[i] }
func (p orderedParameters) Less(i, j int) bool {
	if p[i].Name != p[j].Name {
		return p[i].Name < p[j].Name
	}
	return p[i].Description < p[j].Description
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

func findCommonParameters(routes []restful.Route) (map[interface{}]spec.Parameter, error) {
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
			openAPIParam, err := buildParameter(paramNameKindToDataMap[key])
			if err != nil {
				return commonParamsMap, err
			}
			commonParamsMap[key] = openAPIParam
		}
	}
	return commonParamsMap, nil
}

func buildParameter(restParam restful.ParameterData) (ret spec.Parameter, err error) {
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
		ret.Schema = &spec.Schema{
			SchemaProps: spec.SchemaProps{
				Ref: spec.MustCreateRef("#/definitions/" + restParam.DataType),
			},
		}
		return ret, nil
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
	if !isSimpleDataType(restParam.DataType) {
		return ret, fmt.Errorf("Restful DataType should be a simple type, but got : %v", restParam.DataType)
	}
	ret.Type = restParam.DataType
	ret.Format = restParam.DataFormat
	ret.UniqueItems = !restParam.AllowMultiple
	// TODO(mbohlool): make sure the type of default value matches Type
	if restParam.DefaultValue != "" {
		ret.Default = restParam.DefaultValue
	}

	return ret, nil
}

func buildParameters(restParam []*restful.Parameter) (ret []spec.Parameter, err error) {
	ret = make([]spec.Parameter, len(restParam))
	for i, v := range restParam {
		ret[i], err = buildParameter(v.Data())
		if err != nil {
			return ret, err
		}
	}
	return ret, nil
}

func isSimpleDataType(typeName string) bool {
	switch typeName {
	// Note that "file" intentionally kept out of this list as it is not being used.
	// "file" type has more requirements.
	case "string", "number", "integer", "boolean", "array":
		return true
	}
	return false
}

func getFloat64OrNil(str string) (*float64, error) {
	if len(str) > 0 {
		num, err := strconv.ParseFloat(str, 64)
		return &num, err
	}
	return nil, nil
}

// TODO(mbohlool): Convert default value type to the type of parameter
func getDefaultValue(str swagger.Special) interface{} {
	if len(str) > 0 {
		return str
	}
	return nil
}

func buildType(swaggerType *string, swaggerFormat string) ([]string, string, error) {
	if swaggerType == nil {
		return []string{}, "", nil
	}
	switch *swaggerType {
	case "integer", "number", "string", "boolean", "array", "object", "file":
		return []string{*swaggerType}, swaggerFormat, nil
	case "int":
		return []string{"integer"}, "int32", nil
	case "long":
		return []string{"integer"}, "int64", nil
	case "float", "double":
		return []string{"number"}, *swaggerType, nil
	case "byte", "date", "datetime", "date-time":
		return []string{"string"}, *swaggerType, nil
	default:
		return []string{}, "", fmt.Errorf("Unrecognized swagger 1.2 type : %v, %v", swaggerType, swaggerFormat)
	}
}

// A simple trie implementation with Add an HasPrefix methods only.
type trie struct {
	children map[byte]*trie
}

func createTrie(list []string) trie {
	ret := trie{
		children: make(map[byte]*trie),
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
			child = new(trie)
			child.children = make(map[byte]*trie)
			root.children[b] = child
		}
		root = child
	}
}

func (t *trie) HasPrefix(v string) bool {
	root := t
	for _, b := range []byte(v) {
		child, exists := root.children[b]
		if !exists {
			return false
		}
		root = child
	}
	return true
}
