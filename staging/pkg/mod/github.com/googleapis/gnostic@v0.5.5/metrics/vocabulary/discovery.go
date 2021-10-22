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

package vocabulary

import (
	discovery_v1 "github.com/googleapis/gnostic/discovery"
	metrics "github.com/googleapis/gnostic/metrics"
)

func (vocab *Vocabulary) processMethodDiscovery(operation *discovery_v1.Method) {
	if operation.Id != "" {
		vocab.operationID[operation.Id]++
	}
	if operation.Parameters != nil {
		for _, pair := range operation.Parameters.AdditionalProperties {
			vocab.parameters[pair.Name]++
			vocab.processParameterDiscovery(pair.Value)
		}
	}
}

func (vocab *Vocabulary) processSchemaDiscovery(schema *discovery_v1.Schema) {
	if schema.Properties != nil {
		for _, pair := range schema.Properties.AdditionalProperties {
			vocab.properties[pair.Name]++
			vocab.processSchemaDiscovery(pair.Value)
		}
	}
}

func (vocab *Vocabulary) processParameterDiscovery(parameter *discovery_v1.Parameter) {
	if parameter.Properties != nil {
		for _, pair := range parameter.Properties.AdditionalProperties {
			vocab.properties[pair.Name]++
			vocab.processSchemaDiscovery(pair.Value)
		}
	}
}

func (vocab *Vocabulary) processResourceDiscovery(resource *discovery_v1.Resource) {
	if resource.Methods != nil {
		for _, pair := range resource.Methods.AdditionalProperties {
			vocab.properties[pair.Name]++
			vocab.processMethodDiscovery(pair.Value)
		}
	}
	if resource.Resources != nil {
		for _, pair := range resource.Resources.AdditionalProperties {
			vocab.processResourceDiscovery(pair.Value)
		}
	}
}

// NewVocabularyFromDiscovery collects the vocabulary of a Discovery document.
func NewVocabularyFromDiscovery(document *discovery_v1.Document) *metrics.Vocabulary {
	var vocab Vocabulary
	vocab.schemas = make(map[string]int)
	vocab.operationID = make(map[string]int)
	vocab.parameters = make(map[string]int)
	vocab.properties = make(map[string]int)

	if document.Parameters != nil {
		for _, pair := range document.Parameters.AdditionalProperties {
			vocab.parameters[pair.Name]++
			vocab.processParameterDiscovery(pair.Value)
		}
	}
	if document.Schemas != nil {
		for _, pair := range document.Schemas.AdditionalProperties {
			vocab.schemas[pair.Name]++
			vocab.processSchemaDiscovery(pair.Value)
		}
	}
	if document.Methods != nil {
		for _, pair := range document.Methods.AdditionalProperties {
			vocab.processMethodDiscovery(pair.Value)
		}
	}
	if document.Resources != nil {
		for _, pair := range document.Resources.AdditionalProperties {
			vocab.processResourceDiscovery(pair.Value)
		}
	}
	v := &metrics.Vocabulary{
		Schemas:    fillProtoStructures(vocab.schemas),
		Operations: fillProtoStructures(vocab.operationID),
		Parameters: fillProtoStructures(vocab.parameters),
		Properties: fillProtoStructures(vocab.properties),
	}

	return v
}
