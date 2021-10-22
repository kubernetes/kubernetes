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

	rules "github.com/googleapis/gnostic/metrics/rules"
	pb "github.com/googleapis/gnostic/openapiv2"
)

// fillProtoStructure takes a slice of rules and coverts them to a slice of
// Message structs.
func fillProtoStructure(m []rules.MessageType) []*Message {
	messages := make([]*Message, 0)
	for _, message := range m {
		temp := &Message{
			Type:    message.Message[0],
			Message: message.Message[1],
			Keys:    message.Path,
		}
		if message.Message[2] != "" {
			temp.Suggestion = message.Message[2]
		}
		messages = append(messages, temp)
	}
	return messages
}

// gatherParametersV2 takes a Document struct as a parameter and calls the
// processParamater function on each HTTP request in order to gather the parameters.
func gatherParametersV2(document *pb.Document) []rules.Field {
	p := make([]rules.Field, 0)
	if document.Paths != nil {
		for _, pair := range document.Paths.Path {
			v := pair.Value
			path := []string{"paths", pair.Name}
			if v.Get != nil {
				p = append(p, processParametersV2(v.Get, append(path, "get", "parameters"))...)
			}
			if v.Put != nil {
				p = append(p, processParametersV2(v.Put, append(path, "put", "parameters"))...)
			}
			if v.Post != nil {
				p = append(p, processParametersV2(v.Post, append(path, "post", "parameters"))...)
			}
			if v.Delete != nil {
				p = append(p, processParametersV2(v.Delete, append(path, "delete", "parameters"))...)
			}
			if v.Patch != nil {
				p = append(p, processParametersV2(v.Patch, append(path, "patch", "parameters"))...)
			}
		}
	}
	return p
}

// processParametersV2 loops over the parameters of an operation and creates a
// Field struct slice which will be used for the linter.
func processParametersV2(operation *pb.Operation, path []string) []rules.Field {
	parameters := make([]rules.Field, 0)
	for i, item := range operation.Parameters {
		switch t := item.Oneof.(type) {
		case *pb.ParametersItem_Parameter:
			switch t2 := t.Parameter.Oneof.(type) {
			case *pb.Parameter_BodyParameter:
				parameters = append(parameters, rules.Field{Name: t2.BodyParameter.Name, Path: append(path, fmt.Sprintf("%d", i), "name")})
			case *pb.Parameter_NonBodyParameter:
				switch t3 := t2.NonBodyParameter.Oneof.(type) {
				case *pb.NonBodyParameter_FormDataParameterSubSchema:
					parameters = append(parameters, rules.Field{Name: t3.FormDataParameterSubSchema.Name, Path: append(path, fmt.Sprintf("%d", i), "name")})
				case *pb.NonBodyParameter_HeaderParameterSubSchema:
					parameters = append(parameters, rules.Field{Name: t3.HeaderParameterSubSchema.Name, Path: append(path, fmt.Sprintf("%d", i), "name")})
				case *pb.NonBodyParameter_PathParameterSubSchema:
					parameters = append(parameters, rules.Field{Name: t3.PathParameterSubSchema.Name, Path: append(path, fmt.Sprintf("%d", i), "name")})
				case *pb.NonBodyParameter_QueryParameterSubSchema:
					parameters = append(parameters, rules.Field{Name: t3.QueryParameterSubSchema.Name, Path: append(path, fmt.Sprintf("%d", i), "name")})
				}
			}
		}
	}
	return parameters
}

//AIPLintV2 accepts an OpenAPI v2 document and will call the individual AIP rules
//on the document.
func AIPLintV2(document *pb.Document) (*Linter, int) {
	fields := gatherParametersV2(document)
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
