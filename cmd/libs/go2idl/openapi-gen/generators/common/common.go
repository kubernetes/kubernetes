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

import "github.com/go-openapi/spec"

// OpenAPIDefinition describes single type. Normally these definitions are auto-generated using gen-openapi.
type OpenAPIDefinition struct {
	Schema       spec.Schema
	Dependencies []string
}

// OpenAPIDefinitions is collection of all definitions.
type OpenAPIDefinitions map[string]OpenAPIDefinition

// OpenAPIDefinitionGetter gets openAPI definitions for a given type. If a type implements this interface,
// the definition returns by it will be used, otherwise the auto-generated definitions will be used.
type OpenAPIDefinitionGetter interface {
	OpenAPIDefinition() *OpenAPIDefinition
}

// This function is a reference for converting go (or any custom type) to a simple open API type,format pair
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
