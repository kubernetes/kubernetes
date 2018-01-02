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
	"bytes"
	"encoding/base64"
	"fmt"
	"log"
	_ "os"
	"path"
	"path/filepath"
	"strings"
	"text/template"
	"unicode"
	"unicode/utf8"

	openapi "github.com/googleapis/gnostic/OpenAPIv2"
	plugins "github.com/googleapis/gnostic/plugins"
)

// ServiceType typically corresponds to a definition, parameter,
// or response in the API and is represented by a type in generated code.
type ServiceType struct {
	Name   string
	Kind   string
	Fields []*ServiceTypeField
}

func (s *ServiceType) hasFieldNamed(name string) bool {
	for _, f := range s.Fields {
		if f.Name == name {
			return true
		}
	}
	return false
}

// ServiceTypeField is a field in a definition and can be
// associated with a position in a request structure.
type ServiceTypeField struct {
	Name          string // the name as specified
	Type          string // the specified type of the field
	NativeType    string // the programming-language native type of the field
	FieldName     string // the name to use for data structure fields
	ParameterName string // the name to use for parameters
	JSONName      string // the name to use in JSON serialization
	Position      string // "body", "header", "formdata", "query", or "path"
}

// ServiceMethod is an operation of an API and typically
// has associated client and server code.
type ServiceMethod struct {
	Name               string       // Operation name, possibly generated from method and path
	Path               string       // HTTP path
	Method             string       // HTTP method name
	Description        string       // description of method
	HandlerName        string       // name of the generated handler
	ProcessorName      string       // name of the processing function in the service interface
	ClientName         string       // name of client
	ResultTypeName     string       // native type name for the result structure
	ParametersTypeName string       // native type name for the input parameters structure
	ResponsesTypeName  string       // native type name for the responses
	ParametersType     *ServiceType // parameters (input)
	ResponsesType      *ServiceType // responses (output)
}

// ServiceRenderer reads an OpenAPI document and generates code.
type ServiceRenderer struct {
	Templates map[string]*template.Template

	Name    string
	Package string
	Types   []*ServiceType
	Methods []*ServiceMethod
}

// NewServiceRenderer creates a renderer.
func NewServiceRenderer(document *openapi.Document, packageName string) (renderer *ServiceRenderer, err error) {
	renderer = &ServiceRenderer{}
	// Load templates.
	err = renderer.loadTemplates(templates())
	if err != nil {
		return nil, err
	}
	// Set renderer properties from passed-in document.
	renderer.Name = document.Info.Title
	renderer.Package = packageName // Set package name from argument.
	renderer.Types = make([]*ServiceType, 0)
	renderer.Methods = make([]*ServiceMethod, 0)
	err = renderer.loadService(document)
	if err != nil {
		return nil, err
	}
	return renderer, nil
}

// Load templates that will be used by the renderer.
func (renderer *ServiceRenderer) loadTemplates(files map[string]string) (err error) {
	helpers := templateHelpers()
	renderer.Templates = make(map[string]*template.Template, 0)
	for filename, encoding := range files {
		templateData, err := base64.StdEncoding.DecodeString(encoding)
		if err != nil {
			return err
		}
		t, err := template.New(filename).Funcs(helpers).Parse(string(templateData))
		if err != nil {
			return err
		}
		renderer.Templates[filename] = t
	}
	return err
}

// Preprocess the types and methods of the API.
func (renderer *ServiceRenderer) loadService(document *openapi.Document) (err error) {
	// Collect service type descriptions from Definitions section.
	if document.Definitions != nil {
		for _, pair := range document.Definitions.AdditionalProperties {
			var t ServiceType
			t.Fields = make([]*ServiceTypeField, 0)
			schema := pair.Value
			if schema.Properties != nil {
				if len(schema.Properties.AdditionalProperties) > 0 {
					// If the schema has properties, generate a struct.
					t.Kind = "struct"
				}
				for _, pair2 := range schema.Properties.AdditionalProperties {
					var f ServiceTypeField
					f.Name = strings.Title(pair2.Name)
					f.Type = typeForSchema(pair2.Value)
					f.JSONName = pair2.Name
					t.Fields = append(t.Fields, &f)
				}
			}
			t.Name = strings.Title(filteredTypeName(pair.Name))
			if len(t.Fields) == 0 {
				if schema.AdditionalProperties != nil {
					// If the schema has no fixed properties and additional properties of a specified type,
					// generate a map pointing to objects of that type.
					mapType := typeForRef(schema.AdditionalProperties.GetSchema().XRef)
					t.Kind = "map[string]" + mapType
				}
			}
			renderer.Types = append(renderer.Types, &t)
		}
	}
	// Collect service method descriptions from Paths section.
	for _, pair := range document.Paths.Path {
		v := pair.Value
		if v.Get != nil {
			renderer.loadOperation(v.Get, "GET", pair.Name)
		}
		if v.Post != nil {
			renderer.loadOperation(v.Post, "POST", pair.Name)
		}
		if v.Put != nil {
			renderer.loadOperation(v.Put, "PUT", pair.Name)
		}
		if v.Delete != nil {
			renderer.loadOperation(v.Delete, "DELETE", pair.Name)
		}
	}
	return err
}

// convert the first character of a string to upper case
func upperFirst(s string) string {
	if s == "" {
		return ""
	}
	r, n := utf8.DecodeRuneInString(s)
	return string(unicode.ToUpper(r)) + strings.ToLower(s[n:])
}

func generateOperationName(method, path string) string {

	filteredPath := strings.Replace(path, "/", "_", -1)
	filteredPath = strings.Replace(filteredPath, ".", "_", -1)
	filteredPath = strings.Replace(filteredPath, "{", "", -1)
	filteredPath = strings.Replace(filteredPath, "}", "", -1)

	return upperFirst(method) + filteredPath
}

func (renderer *ServiceRenderer) loadOperation(op *openapi.Operation, method string, path string) (err error) {
	var m ServiceMethod
	m.Name = strings.Title(op.OperationId)
	m.Path = path
	m.Method = method
	if m.Name == "" {
		m.Name = generateOperationName(method, path)
	}
	m.Description = op.Description
	m.HandlerName = "Handle" + m.Name
	m.ProcessorName = m.Name
	m.ClientName = m.Name
	m.ParametersType, err = renderer.loadServiceTypeFromParameters(m.Name+"Parameters", op.Parameters)
	if m.ParametersType != nil {
		m.ParametersTypeName = m.ParametersType.Name
	}
	m.ResponsesType, err = renderer.loadServiceTypeFromResponses(&m, m.Name+"Responses", op.Responses)
	if m.ResponsesType != nil {
		m.ResponsesTypeName = m.ResponsesType.Name
	}
	renderer.Methods = append(renderer.Methods, &m)
	return err
}

func (renderer *ServiceRenderer) loadServiceTypeFromParameters(name string, parameters []*openapi.ParametersItem) (t *ServiceType, err error) {
	t = &ServiceType{}
	t.Kind = "struct"
	t.Fields = make([]*ServiceTypeField, 0)
	for _, parametersItem := range parameters {
		var f ServiceTypeField
		f.Type = fmt.Sprintf("%+v", parametersItem)
		parameter := parametersItem.GetParameter()
		if parameter != nil {
			bodyParameter := parameter.GetBodyParameter()
			if bodyParameter != nil {
				f.Name = bodyParameter.Name
				f.FieldName = strings.Replace(f.Name, "-", "_", -1)
				if bodyParameter.Schema != nil {
					f.Type = typeForSchema(bodyParameter.Schema)
					f.NativeType = f.Type
					f.Position = "body"
				}
			}
			nonBodyParameter := parameter.GetNonBodyParameter()
			if nonBodyParameter != nil {
				headerParameter := nonBodyParameter.GetHeaderParameterSubSchema()
				if headerParameter != nil {
					f.Name = headerParameter.Name
					f.FieldName = strings.Replace(f.Name, "-", "_", -1)
					f.Type = headerParameter.Type
					f.NativeType = f.Type
					f.Position = "header"
				}
				formDataParameter := nonBodyParameter.GetFormDataParameterSubSchema()
				if formDataParameter != nil {
					f.Name = formDataParameter.Name
					f.FieldName = strings.Replace(f.Name, "-", "_", -1)
					f.Type = formDataParameter.Type
					f.NativeType = f.Type
					f.Position = "formdata"
				}
				queryParameter := nonBodyParameter.GetQueryParameterSubSchema()
				if queryParameter != nil {
					f.Name = queryParameter.Name
					f.FieldName = strings.Replace(f.Name, "-", "_", -1)
					f.Type = queryParameter.Type
					f.NativeType = f.Type
					f.Position = "query"
				}
				pathParameter := nonBodyParameter.GetPathParameterSubSchema()
				if pathParameter != nil {
					f.Name = pathParameter.Name
					f.FieldName = strings.Replace(f.Name, "-", "_", -1)
					f.Type = pathParameter.Type
					f.NativeType = f.Type
					f.Position = "path"
					f.Type = typeForName(pathParameter.Type, pathParameter.Format)
				}
			}
			f.JSONName = f.Name
			f.ParameterName = replaceReservedWords(f.FieldName)
			f.Name = strings.Title(f.Name)
			t.Fields = append(t.Fields, &f)
			if f.NativeType == "integer" {
				f.NativeType = "int64"
			}
		}
	}
	t.Name = name
	if len(t.Fields) > 0 {
		renderer.Types = append(renderer.Types, t)
		return t, err
	}
	return nil, err
}

func (renderer *ServiceRenderer) loadServiceTypeFromResponses(m *ServiceMethod, name string, responses *openapi.Responses) (t *ServiceType, err error) {
	t = &ServiceType{}
	t.Kind = "struct"
	t.Fields = make([]*ServiceTypeField, 0)

	for _, responseCode := range responses.ResponseCode {
		var f ServiceTypeField
		f.Name = propertyNameForResponseCode(responseCode.Name)
		f.JSONName = ""
		response := responseCode.Value.GetResponse()
		if response != nil && response.Schema != nil && response.Schema.GetSchema() != nil {
			f.Type = "*" + typeForSchema(response.Schema.GetSchema())
			t.Fields = append(t.Fields, &f)
			if f.Name == "OK" {
				m.ResultTypeName = typeForSchema(response.Schema.GetSchema())
			}
		}
	}

	t.Name = name
	if len(t.Fields) > 0 {
		renderer.Types = append(renderer.Types, t)
		return t, err
	}
	return nil, err
}

func filteredTypeName(typeName string) (name string) {
	// first take the last path segment
	parts := strings.Split(typeName, "/")
	name = parts[len(parts)-1]
	// then take the last part of a dotted name
	parts = strings.Split(name, ".")
	name = parts[len(parts)-1]
	return name
}

func typeForName(name string, format string) (typeName string) {
	switch name {
	case "integer":
		if format == "int32" {
			return "int32"
		} else if format == "int64" {
			return "int64"
		} else {
			return "int32"
		}
	default:
		return name
	}
}

func typeForSchema(schema *openapi.Schema) (typeName string) {
	ref := schema.XRef
	if ref != "" {
		return typeForRef(ref)
	}
	if schema.Type != nil {
		types := schema.Type.Value
		format := schema.Format
		if len(types) == 1 && types[0] == "string" {
			return "string"
		}
		if len(types) == 1 && types[0] == "integer" && format == "int32" {
			return "int32"
		}
		if len(types) == 1 && types[0] == "integer" {
			return "int"
		}
		if len(types) == 1 && types[0] == "number" {
			return "int"
		}
		if len(types) == 1 && types[0] == "array" && schema.Items != nil {
			// we have an array.., but of what?
			items := schema.Items.Schema
			if len(items) == 1 && items[0].XRef != "" {
				return "[]" + typeForRef(items[0].XRef)
			}
		}
		if len(types) == 1 && types[0] == "object" && schema.AdditionalProperties == nil {
			return "map[string]interface{}"
		}
	}
	if schema.AdditionalProperties != nil {
		additionalProperties := schema.AdditionalProperties
		if propertySchema := additionalProperties.GetSchema(); propertySchema != nil {
			if ref := propertySchema.XRef; ref != "" {
				return "map[string]" + typeForRef(ref)
			}
		}
	}
	// this function is incomplete... so return a string representing anything that we don't handle
	return fmt.Sprintf("%v", schema)
}

func typeForRef(ref string) (typeName string) {
	return strings.Replace(strings.Title(path.Base(ref)), "-", "_", -1)
}

func propertyNameForResponseCode(code string) string {
	if code == "200" {
		return "OK"
	}
	return strings.Title(code)
}

// Generate runs the renderer to generate the named files.
func (renderer *ServiceRenderer) Generate(response *plugins.Response, files []string) (err error) {
	for _, filename := range files {
		file := &plugins.File{}
		file.Name = filename
		f := new(bytes.Buffer)
		t := renderer.Templates[filename]
		log.Printf("Generating %s", filename)
		err = t.Execute(f, struct {
			Renderer *ServiceRenderer
		}{
			renderer,
		})
		if err != nil {
			response.Errors = append(response.Errors, fmt.Sprintf("ERROR %v", err))
		}
		inputBytes := f.Bytes()
		// run generated Go files through gofmt
		if filepath.Ext(file.Name) == ".go" {
			strippedBytes := stripMarkers(inputBytes)
			file.Data, err = gofmt(file.Name, strippedBytes)
		} else {
			file.Data = inputBytes
		}
		response.Files = append(response.Files, file)
	}
	return
}

func replaceReservedWords(name string) string {
	log.Printf("replacing %s\n", name)
	if name == "type" {
		return "ttttype"
	}
	return name

}
