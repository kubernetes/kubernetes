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
	"log"
	"sort"

	"sigs.k8s.io/kustomize/pkg/transformers/config/defaultconfig"
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

// MakeEmptyConfig returns an empty TransformerConfig object
func MakeEmptyConfig() *TransformerConfig {
	return &TransformerConfig{}
}

// MakeDefaultConfig returns a default TransformerConfig.
func MakeDefaultConfig() *TransformerConfig {
	c, err := makeTransformerConfigFromBytes(
		defaultconfig.GetDefaultFieldSpecs())
	if err != nil {
		log.Fatalf("Unable to make default transformconfig: %v", err)
	}
	return c
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
func (t *TransformerConfig) AddPrefixFieldSpec(fs FieldSpec) (err error) {
	t.NamePrefix, err = t.NamePrefix.mergeOne(fs)
	return err
}

// AddSuffixFieldSpec adds a FieldSpec to NameSuffix
func (t *TransformerConfig) AddSuffixFieldSpec(fs FieldSpec) (err error) {
	t.NameSuffix, err = t.NameSuffix.mergeOne(fs)
	return err
}

// AddLabelFieldSpec adds a FieldSpec to CommonLabels
func (t *TransformerConfig) AddLabelFieldSpec(fs FieldSpec) (err error) {
	t.CommonLabels, err = t.CommonLabels.mergeOne(fs)
	return err
}

// AddAnnotationFieldSpec adds a FieldSpec to CommonAnnotations
func (t *TransformerConfig) AddAnnotationFieldSpec(fs FieldSpec) (err error) {
	t.CommonAnnotations, err = t.CommonAnnotations.mergeOne(fs)
	return err
}

// AddNamereferenceFieldSpec adds a NameBackReferences to NameReference
func (t *TransformerConfig) AddNamereferenceFieldSpec(
	nbrs NameBackReferences) (err error) {
	t.NameReference, err = t.NameReference.mergeOne(nbrs)
	return err
}

// Merge merges two TransformerConfigs objects into
// a new TransformerConfig object
func (t *TransformerConfig) Merge(input *TransformerConfig) (
	merged *TransformerConfig, err error) {
	if input == nil {
		return t, nil
	}
	merged = &TransformerConfig{}
	merged.NamePrefix, err = t.NamePrefix.mergeAll(input.NamePrefix)
	if err != nil {
		return nil, err
	}
	merged.NameSuffix, err = t.NameSuffix.mergeAll(input.NameSuffix)
	if err != nil {
		return nil, err
	}
	merged.NameSpace, err = t.NameSpace.mergeAll(input.NameSpace)
	if err != nil {
		return nil, err
	}
	merged.CommonAnnotations, err = t.CommonAnnotations.mergeAll(
		input.CommonAnnotations)
	if err != nil {
		return nil, err
	}
	merged.CommonLabels, err = t.CommonLabels.mergeAll(input.CommonLabels)
	if err != nil {
		return nil, err
	}
	merged.VarReference, err = t.VarReference.mergeAll(input.VarReference)
	if err != nil {
		return nil, err
	}
	merged.NameReference, err = t.NameReference.mergeAll(input.NameReference)
	if err != nil {
		return nil, err
	}
	merged.sortFields()
	return merged, nil
}
