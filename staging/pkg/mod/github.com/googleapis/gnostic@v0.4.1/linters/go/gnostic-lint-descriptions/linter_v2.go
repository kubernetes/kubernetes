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
	openapi "github.com/googleapis/gnostic/openapiv2"
	plugins "github.com/googleapis/gnostic/plugins"
)

// DocumentLinter contains information collected about an API description.
type DocumentLinterV2 struct {
	document *openapi.Document `json:"-"`
}

func (d *DocumentLinterV2) Run() []*plugins.Message {
	return d.analyzeDocument(d.document)
}

// NewDocumentLinter builds a new DocumentLinter object.
func NewDocumentLinterV2(document *openapi.Document) *DocumentLinterV2 {
	return &DocumentLinterV2{document: document}
}

// Analyze an OpenAPI description.
func (s *DocumentLinterV2) analyzeDocument(document *openapi.Document) []*plugins.Message {
	messages := make([]*plugins.Message, 0, 0)
	for _, pair := range document.Paths.Path {
		path := pair.Value
		if path.Get != nil {
			messages = append(messages, s.analyzeOperation([]string{"paths", pair.Name, "get"}, path.Get)...)
		}
		if path.Post != nil {
			messages = append(messages, s.analyzeOperation([]string{"paths", pair.Name, "post"}, path.Post)...)
		}
		if path.Put != nil {
			messages = append(messages, s.analyzeOperation([]string{"paths", pair.Name, "put"}, path.Put)...)
		}
		if path.Delete != nil {
			messages = append(messages, s.analyzeOperation([]string{"paths", pair.Name, "delete"}, path.Delete)...)
		}
	}
	if document.Definitions != nil {
		for _, pair := range document.Definitions.AdditionalProperties {
			definition := pair.Value
			messages = append(messages, s.analyzeDefinition([]string{"definitions", pair.Name}, definition)...)
		}
	}
	return messages
}

func (s *DocumentLinterV2) analyzeOperation(keys []string, operation *openapi.Operation) []*plugins.Message {
	messages := make([]*plugins.Message, 0)

	if operation.Description == "" {
		messages = append(messages,
			&plugins.Message{
				Level: plugins.Message_WARNING,
				Code:  "NODESCRIPTION",
				Text:  "Operation has no description.",
				Keys:  keys})
	}
	for _, parameter := range operation.Parameters {
		p := parameter.GetParameter()
		if p != nil {
			b := p.GetBodyParameter()
			if b != nil && b.Description == "" {
				messages = append(messages,
					&plugins.Message{
						Level: plugins.Message_WARNING,
						Code:  "NODESCRIPTION",
						Text:  "Parameter has no description.",
						Keys:  append(keys, []string{"responses", b.Name}...)})
			}
			n := p.GetNonBodyParameter()
			if n != nil {
				hp := n.GetHeaderParameterSubSchema()
				if hp != nil && hp.Description == "" {
					messages = append(messages,
						&plugins.Message{
							Level: plugins.Message_WARNING,
							Code:  "NODESCRIPTION",
							Text:  "Parameter has no description.",
							Keys:  append(keys, []string{"responses", hp.Name}...)})
				}
				fp := n.GetFormDataParameterSubSchema()
				if fp != nil && fp.Description == "" {
					messages = append(messages,
						&plugins.Message{
							Level: plugins.Message_WARNING,
							Code:  "NODESCRIPTION",
							Text:  "Parameter has no description.",
							Keys:  append(keys, []string{"responses", fp.Name}...)})
				}
				qp := n.GetQueryParameterSubSchema()
				if qp != nil && qp.Description == "" {
					messages = append(messages,
						&plugins.Message{
							Level: plugins.Message_WARNING,
							Code:  "NODESCRIPTION",
							Text:  "Parameter has no description.",
							Keys:  append(keys, []string{"responses", qp.Name}...)})
				}
				pp := n.GetPathParameterSubSchema()
				if pp != nil && pp.Description == "" {
					messages = append(messages,
						&plugins.Message{
							Level: plugins.Message_WARNING,
							Code:  "NODESCRIPTION",
							Text:  "Parameter has no description.",
							Keys:  append(keys, []string{"responses", pp.Name}...)})
				}
			}
		}
	}
	for _, pair := range operation.Responses.ResponseCode {
		value := pair.Value
		response := value.GetResponse()
		if response != nil {
			responseSchema := response.Schema
			responseSchemaSchema := responseSchema.GetSchema()
			if responseSchemaSchema != nil && responseSchemaSchema.Description == "" {
				messages = append(messages,
					&plugins.Message{
						Level: plugins.Message_WARNING,
						Code:  "NODESCRIPTION",
						Text:  "Response has no description.",
						Keys:  append(keys, []string{"responses", pair.Name}...)})
			}
			responseFileSchema := responseSchema.GetFileSchema()
			if responseFileSchema != nil && responseFileSchema.Description == "" {
				messages = append(messages,
					&plugins.Message{
						Level: plugins.Message_WARNING,
						Code:  "NODESCRIPTION",
						Text:  "Response has no description.",
						Keys:  append(keys, []string{"responses", pair.Name}...)})
			}
		}
	}
	return messages
}

// Analyze a definition in an OpenAPI description.
func (s *DocumentLinterV2) analyzeDefinition(keys []string, definition *openapi.Schema) []*plugins.Message {
	messages := make([]*plugins.Message, 0)
	if definition.Description == "" {
		messages = append(messages,
			&plugins.Message{
				Level: plugins.Message_WARNING,
				Code:  "NODESCRIPTION",
				Text:  "Definition has no description.",
				Keys:  keys})
	}

	if definition.Properties != nil {
		for _, pair := range definition.Properties.AdditionalProperties {
			propertySchema := pair.Value
			if propertySchema.Description == "" {
				messages = append(messages,
					&plugins.Message{
						Level: plugins.Message_WARNING,
						Code:  "NODESCRIPTION",
						Text:  "Property has no description.",
						Keys:  append(keys, []string{"properties", pair.Name}...)})
			}
		}
	}
	return messages
}
