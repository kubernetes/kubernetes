// Copyright 2017 Google Inc. All Rights Reserved.
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

package main

import (
	"strings"
	"text/template"
)

// This file contains support functions that are passed into template
// evaluation for use within templates.

func hasFieldNamedOK(s *ServiceType) bool {
	return s.hasFieldNamed("OK")
}

func hasFieldNamedDefault(s *ServiceType) bool {
	return s.hasFieldNamed("Default")
}

func hasParameters(m *ServiceMethod) bool {
	return m.ParametersType != nil
}

func hasResponses(m *ServiceMethod) bool {
	return m.ResponsesType != nil
}

func hasPathParameters(m *ServiceMethod) bool {
	for _, field := range m.ParametersType.Fields {
		if field.Position == "path" {
			return true
		}
	}
	return false
}

func hasFormParameters(m *ServiceMethod) bool {
	for _, field := range m.ParametersType.Fields {
		if field.Position == "formdata" {
			return true
		}
	}
	return false
}

func goType(openapiType string) string {
	switch openapiType {
	case "number":
		return "int"
	default:
		return openapiType
	}
}

func parameterList(m *ServiceMethod) string {
	result := ""
	if m.ParametersType != nil {
		for i, field := range m.ParametersType.Fields {
			if i > 0 {
				result += ", "
			}
			result += field.ParameterName + " " + field.NativeType
		}
	}
	return result
}

func bodyParameterName(m *ServiceMethod) string {
	for _, field := range m.ParametersType.Fields {
		if field.Position == "body" {
			return field.JSONName
		}
	}
	return ""
}

func bodyParameterFieldName(m *ServiceMethod) string {
	for _, field := range m.ParametersType.Fields {
		if field.Position == "body" {
			return field.Name
		}
	}
	return ""
}

func commentForText(text string) string {
	result := ""
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if i > 0 {
			result += "\n"
		}
		result += "// " + line
	}
	return result
}

func templateHelpers() template.FuncMap {
	return template.FuncMap{
		"hasFieldNamedOK":        hasFieldNamedOK,
		"hasFieldNamedDefault":   hasFieldNamedDefault,
		"hasParameters":          hasParameters,
		"hasPathParameters":      hasPathParameters,
		"hasFormParameters":      hasFormParameters,
		"hasResponses":           hasResponses,
		"goType":                 goType,
		"parameterList":          parameterList,
		"bodyParameterName":      bodyParameterName,
		"bodyParameterFieldName": bodyParameterFieldName,
		"commentForText":         commentForText,
	}
}
