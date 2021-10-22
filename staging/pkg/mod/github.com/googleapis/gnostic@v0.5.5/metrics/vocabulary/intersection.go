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
	metrics "github.com/googleapis/gnostic/metrics"
)

// mapIntersection finds the intersection between two Vocabularies.
// This function takes a Vocabulary and checks if the words within
// the current Vocabulary already exist within the first Vocabulary.
// If the word exists in both structures it is added to a temp Vocabulary
// which replaces the old Vocabulary.
func (vocab *Vocabulary) mapIntersection(v *metrics.Vocabulary) {
	schemastemp := make(map[string]int)
	operationIDTemp := make(map[string]int)
	parametersTemp := make(map[string]int)
	propertiesTemp := make(map[string]int)
	for _, s := range v.Schemas {
		value, ok := vocab.schemas[s.Word]
		if ok {
			schemastemp[s.Word] += (value + int(s.Count))
		}
	}
	for _, op := range v.Operations {
		value, ok := vocab.operationID[op.Word]
		if ok {
			operationIDTemp[op.Word] += (value + int(op.Count))
		}
	}
	for _, param := range v.Parameters {
		value, ok := vocab.parameters[param.Word]
		if ok {
			parametersTemp[param.Word] += (value + int(param.Count))
		}
	}
	for _, prop := range v.Properties {
		value, ok := vocab.properties[prop.Word]
		if ok {
			propertiesTemp[prop.Word] += (value + int(prop.Count))
		}
	}
	vocab.schemas = schemastemp
	vocab.operationID = operationIDTemp
	vocab.parameters = parametersTemp
	vocab.properties = propertiesTemp
}

// Intersection implements the intersection operation between multiple Vocabularies.
// The function accepts a slice of Vocabularies and returns a single Vocabulary
// struct which that contains words that were found in all of the Vocabularies.
func Intersection(vocabSlices []*metrics.Vocabulary) *metrics.Vocabulary {
	var vocab Vocabulary
	vocab.schemas = make(map[string]int)
	vocab.operationID = make(map[string]int)
	vocab.parameters = make(map[string]int)
	vocab.properties = make(map[string]int)

	vocab.unpackageVocabulary(vocabSlices[0])
	for i := 1; i < len(vocabSlices); i++ {
		vocab.mapIntersection(vocabSlices[i])
	}

	v := &metrics.Vocabulary{
		Schemas:    fillProtoStructures(vocab.schemas),
		Operations: fillProtoStructures(vocab.operationID),
		Parameters: fillProtoStructures(vocab.parameters),
		Properties: fillProtoStructures(vocab.properties),
	}
	return v
}
