/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package config provides the functions to load default or user provided configurations
// for different transformers
package config

import (
	"sort"
)

// TransformerConfig holds the data needed to perform transformations.
type TransformerConfig struct {
	NamePrefix        fsSlice  `json:"namePrefix,omitempty" yaml:"namePrefix,omitempty"`
	NameSuffix        fsSlice  `json:"nameSuffix,omitempty" yaml:"nameSuffix,omitempty"`
	NameSpace         fsSlice  `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	CommonLabels      fsSlice  `json:"commonLabels,omitempty" yaml:"commonLabels,omitempty"`
	CommonAnnotations fsSlice  `json:"commonAnnotations,omitempty" yaml:"commonAnnotations,omitempty"`
	NameReference     nbrSlice `json:"nameReference,omitempty" yaml:"nameReference,omitempty"`
	VarReference      fsSlice  `json:"varReference,omitempty" yaml:"varReference,omitempty"`
}

// sortFields provides determinism in logging, tests, etc.
func (t *TransformerConfig) sortFields() {
	sort.Sort(t.NamePrefix)
	sort.Sort(t.NameSpace)
	sort.Sort(t.CommonLabels)
	sort.Sort(t.CommonAnnotations)
	sort.Sort(t.NameReference)
	sort.Sort(t.VarReference)
}

// AddPrefixFieldSpec adds a FieldSpec to NamePrefix
func (t *TransformerConfig) AddPrefixFieldSpec(fs FieldSpec) {
	t.NamePrefix = append(t.NamePrefix, fs)
}

// AddSuffixFieldSpec adds a FieldSpec to NameSuffix
func (t *TransformerConfig) AddSuffixFieldSpec(fs FieldSpec) {
	t.NameSuffix = append([]FieldSpec{fs}, t.NameSuffix...)
}

// AddLabelFieldSpec adds a FieldSpec to CommonLabels
func (t *TransformerConfig) AddLabelFieldSpec(fs FieldSpec) {
	t.CommonLabels = append(t.CommonLabels, fs)
}

// AddAnnotationFieldSpec adds a FieldSpec to CommonAnnotations
func (t *TransformerConfig) AddAnnotationFieldSpec(fs FieldSpec) {
	t.CommonAnnotations = append(t.CommonAnnotations, fs)
}

// AddNamereferenceFieldSpec adds a NameBackReferences to NameReference
func (t *TransformerConfig) AddNamereferenceFieldSpec(nbrs NameBackReferences) {
	t.NameReference = t.NameReference.mergeOne(nbrs)
}

// Merge merges two TransformerConfigs objects into a new TransformerConfig object
func (t *TransformerConfig) Merge(input *TransformerConfig) *TransformerConfig {
	if input == nil {
		return t
	}
	merged := &TransformerConfig{}
	merged.NamePrefix = append(t.NamePrefix, input.NamePrefix...)
	merged.NameSuffix = append(input.NameSuffix, t.NameSuffix...)
	merged.NameSpace = append(t.NameSpace, input.NameSpace...)
	merged.CommonAnnotations = append(t.CommonAnnotations, input.CommonAnnotations...)
	merged.CommonLabels = append(t.CommonLabels, input.CommonLabels...)
	merged.VarReference = append(t.VarReference, input.VarReference...)
	merged.NameReference = t.NameReference.mergeAll(input.NameReference)
	merged.sortFields()
	return merged
}
