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

package common

import (
	"net/http"

	"golang.org/x/exp/slices"

	"strings"

	"github.com/emicklei/go-restful/v3"

	"k8s.io/kube-openapi/pkg/spec3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

const (
	// TODO: Make this configurable.
	ExtensionPrefix   = "x-kubernetes-"
	ExtensionV2Schema = ExtensionPrefix + "v2-schema"
)

// OpenAPIDefinition describes single type. Normally these definitions are auto-generated using gen-openapi.
type OpenAPIDefinition struct {
	Schema       spec.Schema
	Dependencies []string
}

type ReferenceCallback func(path string) spec.Ref

// GetOpenAPIDefinitions is collection of all definitions.
type GetOpenAPIDefinitions func(ReferenceCallback) map[string]OpenAPIDefinition

// OpenAPIDefinitionGetter gets openAPI definitions for a given type. If a type implements this interface,
// the definition returned by it will be used, otherwise the auto-generated definitions will be used. See
// GetOpenAPITypeFormat for more information about trade-offs of using this interface or GetOpenAPITypeFormat method when
// possible.
type OpenAPIDefinitionGetter interface {
	OpenAPIDefinition() *OpenAPIDefinition
}

type OpenAPIV3DefinitionGetter interface {
	OpenAPIV3Definition() *OpenAPIDefinition
}

type PathHandler interface {
	Handle(path string, handler http.Handler)
}

type PathHandlerByGroupVersion interface {
	Handle(path string, handler http.Handler)
	HandlePrefix(path string, handler http.Handler)
}

// Config is set of configuration for openAPI spec generation.
type Config struct {
	// List of supported protocols such as https, http, etc.
	ProtocolList []string

	// Info is general information about the API.
	Info *spec.Info

	// DefaultResponse will be used if an operation does not have any responses listed. It
	// will show up as ... "responses" : {"default" : $DefaultResponse} in the spec.
	DefaultResponse *spec.Response

	// ResponseDefinitions will be added to "responses" under the top-level swagger object. This is an object
	// that holds responses definitions that can be used across operations. This property does not define
	// global responses for all operations. For more info please refer:
	//     https://github.com/OAI/OpenAPI-Specification/blob/master/versions/2.0.md#fixed-fields
	ResponseDefinitions map[string]spec.Response

	// CommonResponses will be added as a response to all operation specs. This is a good place to add common
	// responses such as authorization failed.
	CommonResponses map[int]spec.Response

	// List of webservice's path prefixes to ignore
	IgnorePrefixes []string

	// OpenAPIDefinitions should provide definition for all models used by routes. Failure to provide this map
	// or any of the models will result in spec generation failure.
	GetDefinitions GetOpenAPIDefinitions

	// Provides the definition for all models used by routes. One of GetDefinitions or Definitions must be defined to generate a spec.
	// This takes precedent over the GetDefinitions function
	Definitions map[string]OpenAPIDefinition

	// GetOperationIDAndTags returns operation id and tags for a restful route. It is an optional function to customize operation IDs.
	//
	// Deprecated: GetOperationIDAndTagsFromRoute should be used instead. This cannot be specified if using the new Route
	// interface set of funcs.
	GetOperationIDAndTags func(r *restful.Route) (string, []string, error)

	// GetOperationIDAndTagsFromRoute returns operation id and tags for a Route. It is an optional function to customize operation IDs.
	GetOperationIDAndTagsFromRoute func(r Route) (string, []string, error)

	// GetDefinitionName returns a friendly name for a definition base on the serving path. parameter `name` is the full name of the definition.
	// It is an optional function to customize model names.
	GetDefinitionName func(name string) (string, spec.Extensions)

	// PostProcessSpec runs after the spec is ready to serve. It allows a final modification to the spec before serving.
	PostProcessSpec func(*spec.Swagger) (*spec.Swagger, error)

	// SecurityDefinitions is list of all security definitions for OpenAPI service. If this is not nil, the user of config
	// is responsible to provide DefaultSecurity and (maybe) add unauthorized response to CommonResponses.
	SecurityDefinitions *spec.SecurityDefinitions

	// DefaultSecurity for all operations. This will pass as spec.SwaggerProps.Security to OpenAPI.
	// For most cases, this will be list of acceptable definitions in SecurityDefinitions.
	DefaultSecurity []map[string][]string
}

// OpenAPIV3Config is set of configuration for OpenAPI V3 spec generation.
type OpenAPIV3Config struct {
	// Info is general information about the API.
	Info *spec.Info

	// DefaultResponse will be used if an operation does not have any responses listed. It
	// will show up as ... "responses" : {"default" : $DefaultResponse} in the spec.
	DefaultResponse *spec3.Response

	// ResponseDefinitions will be added to responses component. This is an object
	// that holds responses that can be used across operations.
	ResponseDefinitions map[string]*spec3.Response

	// CommonResponses will be added as a response to all operation specs. This is a good place to add common
	// responses such as authorization failed.
	CommonResponses map[int]*spec3.Response

	// List of webservice's path prefixes to ignore
	IgnorePrefixes []string

	// OpenAPIDefinitions should provide definition for all models used by routes. Failure to provide this map
	// or any of the models will result in spec generation failure.
	// One of GetDefinitions or Definitions must be defined to generate a spec.
	GetDefinitions GetOpenAPIDefinitions

	// Provides the definition for all models used by routes. One of GetDefinitions or Definitions must be defined to generate a spec.
	// This takes precedent over the GetDefinitions function
	Definitions map[string]OpenAPIDefinition

	// GetOperationIDAndTags returns operation id and tags for a restful route. It is an optional function to customize operation IDs.
	//
	// Deprecated: GetOperationIDAndTagsFromRoute should be used instead. This cannot be specified if using the new Route
	// interface set of funcs.
	GetOperationIDAndTags func(r *restful.Route) (string, []string, error)

	// GetOperationIDAndTagsFromRoute returns operation id and tags for a Route. It is an optional function to customize operation IDs.
	GetOperationIDAndTagsFromRoute func(r Route) (string, []string, error)

	// GetDefinitionName returns a friendly name for a definition base on the serving path. parameter `name` is the full name of the definition.
	// It is an optional function to customize model names.
	GetDefinitionName func(name string) (string, spec.Extensions)

	// PostProcessSpec runs after the spec is ready to serve. It allows a final modification to the spec before serving.
	PostProcessSpec func(*spec3.OpenAPI) (*spec3.OpenAPI, error)

	// SecuritySchemes is list of all security schemes for OpenAPI service.
	SecuritySchemes spec3.SecuritySchemes

	// DefaultSecurity for all operations.
	DefaultSecurity []map[string][]string
}

type typeInfo struct {
	name   string
	format string
	zero   interface{}
}

var schemaTypeFormatMap = map[string]typeInfo{
	"uint":        {"integer", "int32", 0.},
	"uint8":       {"integer", "byte", 0.},
	"uint16":      {"integer", "int32", 0.},
	"uint32":      {"integer", "int64", 0.},
	"uint64":      {"integer", "int64", 0.},
	"int":         {"integer", "int32", 0.},
	"int8":        {"integer", "byte", 0.},
	"int16":       {"integer", "int32", 0.},
	"int32":       {"integer", "int32", 0.},
	"int64":       {"integer", "int64", 0.},
	"byte":        {"integer", "byte", 0},
	"float64":     {"number", "double", 0.},
	"float32":     {"number", "float", 0.},
	"bool":        {"boolean", "", false},
	"time.Time":   {"string", "date-time", ""},
	"string":      {"string", "", ""},
	"integer":     {"integer", "", 0.},
	"number":      {"number", "", 0.},
	"boolean":     {"boolean", "", false},
	"[]byte":      {"string", "byte", ""}, // base64 encoded characters
	"interface{}": {"object", "", interface{}(nil)},
}

// This function is a reference for converting go (or any custom type) to a simple open API type,format pair. There are
// two ways to customize spec for a type. If you add it here, a type will be converted to a simple type and the type
// comment (the comment that is added before type definition) will be lost. The spec will still have the property
// comment. The second way is to implement OpenAPIDefinitionGetter interface. That function can customize the spec (so
// the spec does not need to be simple type,format) or can even return a simple type,format (e.g. IntOrString). For simple
// type formats, the benefit of adding OpenAPIDefinitionGetter interface is to keep both type and property documentation.
// Example:
//
//	type Sample struct {
//	     ...
//	     // port of the server
//	     port IntOrString
//	     ...
//	}
//
// // IntOrString documentation...
// type IntOrString { ... }
//
// Adding IntOrString to this function:
//
//	"port" : {
//	          format:      "string",
//	          type:        "int-or-string",
//	          Description: "port of the server"
//	}
//
// Implement OpenAPIDefinitionGetter for IntOrString:
//
//	"port" : {
//	          $Ref:    "#/definitions/IntOrString"
//	          Description: "port of the server"
//	}
//
// ...
// definitions:
//
//	{
//	          "IntOrString": {
//	                    format:      "string",
//	                    type:        "int-or-string",
//	                    Description: "IntOrString documentation..."    // new
//	          }
//	}
func OpenAPITypeFormat(typeName string) (string, string) {
	mapped, ok := schemaTypeFormatMap[typeName]
	if !ok {
		return "", ""
	}
	return mapped.name, mapped.format
}

// Returns the zero-value for the given type along with true if the type
// could be found.
func OpenAPIZeroValue(typeName string) (interface{}, bool) {
	mapped, ok := schemaTypeFormatMap[typeName]
	if !ok {
		return nil, false
	}
	return mapped.zero, true
}

func EscapeJsonPointer(p string) string {
	// Escaping reference name using rfc6901
	p = strings.Replace(p, "~", "~0", -1)
	p = strings.Replace(p, "/", "~1", -1)
	return p
}

func EmbedOpenAPIDefinitionIntoV2Extension(main OpenAPIDefinition, embedded OpenAPIDefinition) OpenAPIDefinition {
	if main.Schema.Extensions == nil {
		main.Schema.Extensions = make(map[string]interface{})
	}
	main.Schema.Extensions[ExtensionV2Schema] = embedded.Schema
	return main
}

// GenerateOpenAPIV3OneOfSchema generate the set of schemas that MUST be assigned to SchemaProps.OneOf
func GenerateOpenAPIV3OneOfSchema(types []string) (oneOf []spec.Schema) {
	for _, t := range types {
		oneOf = append(oneOf, spec.Schema{SchemaProps: spec.SchemaProps{Type: []string{t}}})
	}
	return
}

// MaybePopulateIntOrString populates the x-int-or-string extension only if
// the oneOf type refers to supporting both an int (integer/number) and string.
func MaybePopulateIntOrString(oneOf []string, ext spec.Extensions) spec.Extensions {
	if len(oneOf) != 2 {
		return ext
	}
	if slices.Contains(oneOf, "string") && (slices.Contains(oneOf, "integer") || slices.Contains(oneOf, "number")) {
		ext["x-kubernetes-int-or-string"] = true
	}
	return ext
}
