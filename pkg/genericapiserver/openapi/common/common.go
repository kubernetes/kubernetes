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
	"reflect"
	"strings"

	"github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
)

// OpenAPIDefinition describes single type. Normally these definitions are auto-generated using gen-openapi.
type OpenAPIDefinition struct {
	Schema       spec.Schema
	Dependencies []string
}

// OpenAPIDefinitions is collection of all definitions.
type OpenAPIDefinitions map[string]OpenAPIDefinition

// OpenAPIDefinitionGetter gets openAPI definitions for a given type. If a type implements this interface,
// the definition returned by it will be used, otherwise the auto-generated definitions will be used. See
// GetOpenAPITypeFormat for more information about trade-offs of using this interface or GetOpenAPITypeFormat method when
// possible.
type OpenAPIDefinitionGetter interface {
	OpenAPIDefinition() *OpenAPIDefinition
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

	// CommonResponses will be added as a response to all operation specs. This is a good place to add common
	// responses such as authorization failed.
	CommonResponses map[int]spec.Response

	// List of webservice's path prefixes to ignore
	IgnorePrefixes []string

	// OpenAPIDefinitions should provide definition for all models used by routes. Failure to provide this map
	// or any of the models will result in spec generation failure.
	Definitions *OpenAPIDefinitions

	// GetOperationIDAndTags returns operation id and tags for a restful route. It is an optional function to customize operation IDs.
	GetOperationIDAndTags func(servePath string, r *restful.Route) (string, []string, error)

	// SecurityDefinitions is list of all security definitions for OpenAPI service. If this is not nil, the user of config
	// is responsible to provide DefaultSecurity and (maybe) add unauthorized response to CommonResponses.
	SecurityDefinitions *spec.SecurityDefinitions

	// DefaultSecurity for all operations. This will pass as spec.SwaggerProps.Security to OpenAPI.
	// For most cases, this will be list of acceptable definitions in SecurityDefinitions.
	DefaultSecurity []map[string][]string
}

// This function is a reference for converting go (or any custom type) to a simple open API type,format pair. There are
// two ways to customize spec for a type. If you add it here, a type will be converted to a simple type and the type
// comment (the comment that is added before type definition) will be lost. The spec will still have the property
// comment. The second way is to implement OpenAPIDefinitionGetter interface. That function can customize the spec (so
// the spec does not need to be simple type,format) or can even return a simple type,format (e.g. IntOrString). For simple
// type formats, the benefit of adding OpenAPIDefinitionGetter interface is to keep both type and property documentation.
// Example:
// type Sample struct {
//      ...
//      // port of the server
//      port IntOrString
//      ...
// }
// // IntOrString documentation...
// type IntOrString { ... }
//
// Adding IntOrString to this function:
// "port" : {
//           format:      "string",
//           type:        "int-or-string",
//           Description: "port of the server"
// }
//
// Implement OpenAPIDefinitionGetter for IntOrString:
//
// "port" : {
//           $Ref:    "#/definitions/IntOrString"
//           Description: "port of the server"
// }
// ...
// definitions:
// {
//           "IntOrString": {
//                     format:      "string",
//                     type:        "int-or-string",
//                     Description: "IntOrString documentation..."    // new
//           }
// }
//
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
		"integer":   {"integer", ""},
		"number":    {"number", ""},
		"boolean":   {"boolean", ""},
		"[]byte":    {"string", "byte"}, // base64 encoded characters
	}
	mapped, ok := schemaTypeFormatMap[typeName]
	if !ok {
		return "", ""
	}
	return mapped[0], mapped[1]
}

// golangTypeNameOpenAPIVendorExtension is an OpenAPI vendor extension that identifies the Golang
// type name of a given OpenAPI struct.
const golangTypeNameOpenAPIVendorExtension = "io.k8s.kubernetes.openapi.type.golang"

// TypeNameFunction returns a function that can map a given Go type to an OpenAPI name.
func (c *Config) TypeNameFunction() func(t reflect.Type) string {
	typesToDefinitions := make(typeMapper)
	if c != nil && c.Definitions != nil {
		for name, definition := range *c.Definitions {
			if t, ok := definition.Schema.VendorExtensible.Extensions.GetString(golangTypeNameOpenAPIVendorExtension); ok {
				typesToDefinitions[t] = name
			}
		}
	}
	return typesToDefinitions.Name
}

// ObjectTypeNameFunction returns the appropriate name for an object.
func (c *Config) ObjectTypeNameFunction() func(obj interface{}) string {
	fn := c.TypeNameFunction()
	return func(obj interface{}) string { return fn(reflect.TypeOf(obj)) }
}

// typeMapper is a map of Go types names to OpenAPI definition names.
type typeMapper map[string]string

// Name returns the appropriate OpenAPI definition name for a given Go type.
func (m typeMapper) Name(t reflect.Type) string {
	if t.Kind() == reflect.Ptr {
		t = t.Elem()
	}

	typeName := t.Name()
	if len(t.PkgPath()) > 0 {
		typeName = t.PkgPath() + "." + typeName
	}

	// previously registered types
	if name, ok := m[typeName]; ok {
		return name
	}

	// unknown types
	name := t.PkgPath()
	dirs := strings.Split(name, "/")
	// convert any segments that have dots into reverse domain notation
	for i, s := range dirs {
		if strings.Contains(s, ".") {
			segments := strings.Split(s, ".")
			for j := 0; j < len(segments)/2; j++ {
				end := len(segments) - j - 1
				segments[j], segments[end] = segments[end], segments[j]
			}
			dirs[i] = strings.Join(segments, ".")
		}
	}
	name = strings.Join(dirs, ".")
	name = strings.Replace(name, "-", "_", -1)
	name = strings.ToLower(name)
	if len(name) > 0 {
		name = name + "." + t.Name()
	} else {
		name = t.Name()
	}
	return name
}
