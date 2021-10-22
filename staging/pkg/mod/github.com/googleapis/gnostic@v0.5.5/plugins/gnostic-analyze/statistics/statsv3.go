// Copyright 2017 Google LLC. All Rights Reserved.
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
	openapi "github.com/googleapis/gnostic/openapiv3"
)

// NewDocumentStatistics builds a new DocumentStatistics object.
func NewDocumentStatisticsV3(source string, document *openapi.Document) *DocumentStatistics {
	s := &DocumentStatistics{}
	s.Operations = make(map[string]int, 0)
	s.ParameterTypes = make(map[string]int, 0)
	s.ResultTypes = make(map[string]int, 0)
	s.DefinitionFieldTypes = make(map[string]int, 0)
	s.DefinitionArrayTypes = make(map[string]int, 0)
	s.DefinitionPrimitiveTypes = make(map[string]int, 0)
	s.AnonymousOperations = make([]string, 0)
	s.AnonymousObjects = make([]string, 0)
	// TODO
	//s.analyzeDocumentV3(source, document)
	return s
}

/*
func (s *DocumentStatistics) analyzeOperationV3(method string, path string, operation *openapi.Operation) {
	s.addOperation(method)
	s.addOperation("total")
	if operation.OperationId == "" {
		s.addOperation("anonymous")
		s.AnonymousOperations = append(s.AnonymousOperations, path)
	}
	for _, parametersItem := range operation.Parameters {
		p := parametersItem.GetParameter()
		if p != nil {
			typeName := typeNameForSchemaOrReferenceV3(p.Schema)
			s.addParameterType(path+"/"+p.Name, typeName)
		}
	}

	for _, pair := range *(operation.Responses.Responses) {
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
func (s *DocumentStatistics) analyzeDefinitionV3(path string, definition *openapi.Schema) {
	s.DefinitionCount++
	typeName := typeNameForSchemaV3(definition)
	switch typeName {
	case "object":
		if definition.Properties != nil {
			for _, pair := range definition.Properties.AdditionalProperties {
				propertySchema := pair.Value
				propertyType := typeForSchemaV3(propertySchema)
				s.addDefinitionFieldType(path+"/"+pair.Name, propertyType)
			}
		}
	case "array":
		s.addDefinitionArrayType(path+"/", typeForSchemaV3(definition))
	default: // string, boolean, integer, number, null...
		s.addDefinitionPrimitiveType(path+"/", typeName)
	}
}

// Analyze an OpenAPI description.
// Collect information about types used in the API.
// This should be called exactly once per DocumentStatistics object.
func (s *DocumentStatistics) analyzeDocumentV3(source string, document *openapi.Document) {
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
	if document.Components.Schemas != nil {
		for _, pair := range document.Components.Schemas.AdditionalProperties {
			definition := pair.Value
			if definition.GetSchema() != nil {
				s.analyzeDefinition("definitions/"+pair.Name, definition.GetSchema())
			}
		}
	}
}
*/
