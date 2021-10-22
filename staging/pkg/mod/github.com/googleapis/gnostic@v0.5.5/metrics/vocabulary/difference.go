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

// mapDifference finds the difference between two Vocabularies.
// This function takes a Vocabulary and checks if the words within
// the current Vocabulary already exist within the first Vocabulary.
// If the word exists in both structures it is removed from the
// Vocabulary structure.
func (vocab *Vocabulary) mapDifference(v *metrics.Vocabulary) {
	for _, s := range v.Schemas {
		_, ok := vocab.schemas[s.Word]
		if ok {
			delete(vocab.schemas, s.Word)
		}
	}
	for _, op := range v.Operations {
		_, ok := vocab.operationID[op.Word]
		if ok {
			delete(vocab.operationID, op.Word)
		}
	}
	for _, param := range v.Parameters {
		_, ok := vocab.parameters[param.Word]
		if ok {
			delete(vocab.parameters, param.Word)
		}
	}
	for _, prop := range v.Properties {
		_, ok := vocab.properties[prop.Word]
		if ok {
			delete(vocab.properties, prop.Word)
		}
	}
}

// Difference implements the difference operation between multiple Vocabularies.
// The function accepts a slice of Vocabularies and returns a single Vocabulary
// struct which that contains words that were unique to the first Vocabulary in the slice.
func Difference(vocabularies []*metrics.Vocabulary) *metrics.Vocabulary {
	var vocab Vocabulary
	vocab.schemas = make(map[string]int)
	vocab.operationID = make(map[string]int)
	vocab.parameters = make(map[string]int)
	vocab.properties = make(map[string]int)

	vocab.unpackageVocabulary(vocabularies[0])
	for i := 1; i < len(vocabularies); i++ {
		vocab.mapDifference(vocabularies[i])
	}

	v := &metrics.Vocabulary{
		Schemas:    fillProtoStructures(vocab.schemas),
		Operations: fillProtoStructures(vocab.operationID),
		Parameters: fillProtoStructures(vocab.parameters),
		Properties: fillProtoStructures(vocab.properties),
	}
	return v
}
