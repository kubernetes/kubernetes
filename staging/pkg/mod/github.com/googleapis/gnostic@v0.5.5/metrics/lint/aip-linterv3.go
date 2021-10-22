// Copyright 2020 Google LLC. All Rights Reserved.
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

package linter

import (
	"fmt"
	"os"

	rules "github.com/googleapis/gnostic/metrics/rules"
	openapi_v3 "github.com/googleapis/gnostic/openapiv3"
)

// processParametersV2 loops over the parameters of component and creates a
// Field struct slice which will be used for the linter.
func processParametersV3(components *openapi_v3.Components, path []string) []rules.Field {
	parameters := make([]rules.Field, 0)
	if components.Parameters != nil {
		for _, pair := range components.Parameters.AdditionalProperties {
			switch t := pair.Value.Oneof.(type) {
			case *openapi_v3.ParameterOrReference_Parameter:
				parameters = append(parameters, rules.Field{Name: t.Parameter.Name, Path: append(path, pair.Name, "name")})

			}
		}
	}
	return parameters
}

// processParametersV2 loops over the parameters of an operation and creates a
// Field struct slice which will be used for the linter.
func processOperationV3(operation *openapi_v3.Operation, path []string) []rules.Field {
	parameters := make([]rules.Field, 0)
	for _, item := range operation.Parameters {
		fmt.Fprintf(os.Stderr, "%+v\n", item)
		switch t := item.Oneof.(type) {
		case *openapi_v3.ParameterOrReference_Parameter:
			parameters = append(parameters, rules.Field{Name: t.Parameter.Name, Path: path})

		}
	}
	return parameters
}

// gatherParametersV2 takes a Document struct as a parameter and calls the
// processParamater function on components and each HTTP request in order
// to gather the parameters.
func gatherParameters(document *openapi_v3.Document) []rules.Field {
	p := make([]rules.Field, 0)

	if document.Components != nil {
		path := []string{"components", "parameters"}
		p = append(p, processParametersV3(document.Components, path)...)
	}

	if document.Paths != nil {
		for _, pair := range document.Paths.Path {
			fmt.Fprintf(os.Stderr, "%+v\n", pair)
			v := pair.Value
			path := []string{"paths", pair.Name}
			if v.Get != nil {
				p = append(p, processOperationV3(v.Get, append(path, "get", "parameters", "name"))...)
			}
			if v.Post != nil {
				p = append(p, processOperationV3(v.Post, append(path, "post", "parameters", "name"))...)
			}
			if v.Put != nil {
				p = append(p, processOperationV3(v.Put, append(path, "put", "parameters", "name"))...)
			}
			if v.Patch != nil {
				p = append(p, processOperationV3(v.Patch, append(path, "patch", "parameters", "name"))...)
			}
			if v.Delete != nil {
				p = append(p, processOperationV3(v.Delete, append(path, "delete", "parameters", "name"))...)
			}
		}
	}
	return p
}

//AIPLintV3 accepts an OpenAPI v2 document and will call the individual AIP rules
//on the document.
func AIPLintV3(document *openapi_v3.Document) (*Linter, int) {
	fields := gatherParameters(document)
	messages := make([]rules.MessageType, 0)
	for _, field := range fields {
		messages = append(messages, rules.AIP122Driver(field)...)
		messages = append(messages, rules.AIP140Driver(field)...)
	}
	m := fillProtoStructure(messages)

	linterResult := &Linter{
		Messages: m,
	}
	return linterResult, len(messages)
}
