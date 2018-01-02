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

package statistics

import (
	"fmt"
	"strings"

	openapi "github.com/googleapis/gnostic/OpenAPIv2"
)

// DocumentStatistics contains information collected about an API description.
type DocumentStatistics struct {
	Name                     string         `json:"name"`
	Title                    string         `json:"title"`
	Operations               map[string]int `json:"operations"`
	DefinitionCount          int            `json:"definitions"`
	ParameterTypes           map[string]int `json:"parameterTypes"`
	ResultTypes              map[string]int `json:"resultTypes"`
	DefinitionFieldTypes     map[string]int `json:"definitionFieldTypes"`
	DefinitionArrayTypes     map[string]int `json:"definitionArrayTypes"`
	DefinitionPrimitiveTypes map[string]int `json:"definitionPrimitiveTypes"`
	AnonymousOperations      []string       `json:"anonymousOperations"`
	AnonymousObjects         []string       `json:"anonymousObjects"`
}

// NewDocumentStatistics builds a new DocumentStatistics object.
func NewDocumentStatistics(source string, document *openapi.Document) *DocumentStatistics {
	s := &DocumentStatistics{}
	s.Operations = make(map[string]int, 0)
	s.ParameterTypes = make(map[string]int, 0)
	s.ResultTypes = make(map[string]int, 0)
	s.DefinitionFieldTypes = make(map[string]int, 0)
	s.DefinitionArrayTypes = make(map[string]int, 0)
	s.DefinitionPrimitiveTypes = make(map[string]int, 0)
	s.AnonymousOperations = make([]string, 0)
	s.AnonymousObjects = make([]string, 0)
	s.analyzeDocument(source, document)
	return s
}

func (s *DocumentStatistics) addOperation(name string) {
	s.Operations[name] = s.Operations[name] + 1
}

func (s *DocumentStatistics) addParameterType(path string, name string) {
	if strings.Contains(name, "object") {
		s.AnonymousObjects = append(s.AnonymousObjects, path)
	}
	s.ParameterTypes[name] = s.ParameterTypes[name] + 1
}

func (s *DocumentStatistics) addResultType(path string, name string) {
	if strings.Contains(name, "object") {
		s.AnonymousObjects = append(s.AnonymousObjects, path)
	}
	s.ResultTypes[name] = s.ResultTypes[name] + 1
}

func (s *DocumentStatistics) addDefinitionFieldType(path string, name string) {
	if strings.Contains(name, "object") {
		s.AnonymousObjects = append(s.AnonymousObjects, path)
	}
	s.DefinitionFieldTypes[name] = s.DefinitionFieldTypes[name] + 1
}

func (s *DocumentStatistics) addDefinitionArrayType(path string, name string) {
	if strings.Contains(name, "object") {
		s.AnonymousObjects = append(s.AnonymousObjects, path)
	}
	s.DefinitionArrayTypes[name] = s.DefinitionArrayTypes[name] + 1
}

func (s *DocumentStatistics) addDefinitionPrimitiveType(path string, name string) {
	s.DefinitionPrimitiveTypes[name] = s.DefinitionPrimitiveTypes[name] + 1
}

func typeForPrimitivesItems(p *openapi.PrimitivesItems) string {
	switch {
	case p == nil:
		return "object"
	case p.Type != "":
		return p.Type
	case p.Items != nil && p.Items.Type != "":
		return p.Items.Type
	default:
		return "object"
	}
}

func (s *DocumentStatistics) analyzeOperation(method string, path string, operation *openapi.Operation) {
	s.addOperation(method)
	s.addOperation("total")
	if operation.OperationId == "" {
		s.addOperation("anonymous")
		s.AnonymousOperations = append(s.AnonymousOperations, path)
	}
	for _, parameter := range operation.Parameters {
		p := parameter.GetParameter()
		if p != nil {
			b := p.GetBodyParameter()
			if b != nil {
				typeName := typeForSchema(b.Schema)
				s.addParameterType(path+"/"+b.Name, typeName)
			}
			n := p.GetNonBodyParameter()
			if n != nil {
				hp := n.GetHeaderParameterSubSchema()
				if hp != nil {
					t := hp.Type
					if t == "array" {
						t += "-of-" + typeForPrimitivesItems(hp.Items)
					}
					s.addParameterType(path+"/"+hp.Name, t)
				}
				fp := n.GetFormDataParameterSubSchema()
				if fp != nil {
					t := fp.Type
					if t == "array" {
						t += "-of-" + typeForPrimitivesItems(fp.Items)
					}
					s.addParameterType(path+"/"+fp.Name, t)
				}
				qp := n.GetQueryParameterSubSchema()
				if qp != nil {
					t := qp.Type
					if t == "array" {
						t += "-of-" + typeForPrimitivesItems(qp.Items)
					}
					s.addParameterType(path+"/"+qp.Name, t)
				}
				pp := n.GetPathParameterSubSchema()
				if pp != nil {
					t := pp.Type
					if t == "array" {
						if t == "array" {
							t += "-of-" + typeForPrimitivesItems(pp.Items)
						}
					}
					s.addParameterType(path+"/"+pp.Name, t)
				}
			}
		}
		r := parameter.GetJsonReference()
		if r != nil {
			s.addParameterType(path+"/", "reference")
		}
	}

	for _, pair := range operation.Responses.ResponseCode {
		value := pair.Value
		response := value.GetResponse()
		if response != nil {
			responseSchema := response.Schema
			responseSchemaSchema := responseSchema.GetSchema()
			if responseSchemaSchema != nil {
				s.addResultType(path+"/responses/"+pair.Name, typeForSchema(responseSchemaSchema))
			}
			responseFileSchema := responseSchema.GetFileSchema()
			if responseFileSchema != nil {
				s.addResultType(path+"/responses/"+pair.Name, typeForFileSchema(responseFileSchema))
			}
		}
		ref := value.GetJsonReference()
		if ref != nil {
		}
	}

}

// Analyze a definition in an OpenAPI description.
// Collect information about the definition type and any subsidiary types,
// such as the types of object fields or array elements.
func (s *DocumentStatistics) analyzeDefinition(path string, definition *openapi.Schema) {
	s.DefinitionCount++
	typeName := typeNameForSchema(definition)
	switch typeName {
	case "object":
		if definition.Properties != nil {
			for _, pair := range definition.Properties.AdditionalProperties {
				propertySchema := pair.Value
				propertyType := typeForSchema(propertySchema)
				s.addDefinitionFieldType(path+"/"+pair.Name, propertyType)
			}
		}
	case "array":
		s.addDefinitionArrayType(path+"/", typeForSchema(definition))
	default: // string, boolean, integer, number, null...
		s.addDefinitionPrimitiveType(path+"/", typeName)
	}
}

// Analyze an OpenAPI description.
// Collect information about types used in the API.
// This should be called exactly once per DocumentStatistics object.
func (s *DocumentStatistics) analyzeDocument(source string, document *openapi.Document) {
	s.Name = source

	s.Title = document.Info.Title
	for _, pair := range document.Paths.Path {
		path := pair.Value
		if path.Get != nil {
			s.analyzeOperation("get", "paths"+pair.Name+"/get", path.Get)
		}
		if path.Post != nil {
			s.analyzeOperation("post", "paths"+pair.Name+"/post", path.Post)
		}
		if path.Put != nil {
			s.analyzeOperation("put", "paths"+pair.Name+"/put", path.Put)
		}
		if path.Delete != nil {
			s.analyzeOperation("delete", "paths"+pair.Name+"/delete", path.Delete)
		}
	}
	if document.Definitions != nil {
		for _, pair := range document.Definitions.AdditionalProperties {
			definition := pair.Value
			s.analyzeDefinition("definitions/"+pair.Name, definition)
		}
	}
}

// helpers

func typeNameForSchema(schema *openapi.Schema) string {
	typeName := "object" // default type
	if schema.Type != nil && len(schema.Type.Value) > 0 {
		typeName = ""
		for i, name := range schema.Type.Value {
			if i > 0 {
				typeName += "|"
			}
			typeName += name
		}
	}
	return typeName
}

// Return a type name to use for a schema.
func typeForSchema(schema *openapi.Schema) string {
	if schema.XRef != "" {
		return "reference"
	}
	if len(schema.Enum) > 0 {
		enumType := typeNameForSchema(schema)
		return "enum-of-" + enumType
	}
	typeName := typeNameForSchema(schema)
	if typeName == "array" {
		if schema.Items != nil {
			// items contains an array of schemas
			itemType := ""
			for i, itemSchema := range schema.Items.Schema {
				if i > 0 {
					itemType += "|"
				}
				itemType += typeForSchema(itemSchema)
			}
			return "array-of-" + itemType
		} else if schema.XRef != "" {
			return "array-of-reference"
		} else {
			// we need to do more work to understand this type
			return fmt.Sprintf("array-of-[%+v]", schema)
		}
	} else if typeName == "object" {
		// this object might be representable with a map
		// but not if it has properties
		if (schema.Properties != nil) && (len(schema.Properties.AdditionalProperties) > 0) {
			return typeName
		}
		if schema.AdditionalProperties != nil {
			if schema.AdditionalProperties.GetSchema() != nil {
				additionalPropertiesSchemaType := typeForSchema(schema.AdditionalProperties.GetSchema())
				return "map-of-" + additionalPropertiesSchemaType
			}
			if schema.AdditionalProperties.GetBoolean() == false {
				// no additional properties are allowed, so we're not sure what to do if we get here...
				return typeName
			}
		}
		if schema.Items != nil {
			itemType := ""
			for i, itemSchema := range schema.Items.Schema {
				if i > 0 {
					itemType += "|"
				}
				itemType += typeForSchema(itemSchema)
			}
			return "map-of-" + itemType
		}
		return "map-of-object"
	} else {
		return typeName
	}
}

func typeForFileSchema(schema *openapi.FileSchema) string {
	if schema.Type != "" {
		value := schema.Type
		switch value {
		case "boolean":
			return "fileschema-" + value
		case "string":
			return "fileschema-" + value
		case "file":
			return "fileschema-" + value
		case "number":
			return "fileschema-" + value
		case "integer":
			return "fileschema-" + value
		case "object":
			return "fileschema-" + value
		case "null":
			return "fileschema-" + value
		}
	}
	return fmt.Sprintf("FILE SCHEMA %+v", schema)
}
