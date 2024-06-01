// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinconfig

import (
	"log"
	"sort"
	"sync"

	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/kustomize/api/internal/konfig/builtinpluginconsts"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/errors"
)

// TransformerConfig holds the data needed to perform transformations.
type TransformerConfig struct {
	// if any fields are added, update the DeepCopy implementation
	NamePrefix        types.FsSlice `json:"namePrefix,omitempty" yaml:"namePrefix,omitempty"`
	NameSuffix        types.FsSlice `json:"nameSuffix,omitempty" yaml:"nameSuffix,omitempty"`
	NameSpace         types.FsSlice `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	CommonLabels      types.FsSlice `json:"commonLabels,omitempty" yaml:"commonLabels,omitempty"`
	Labels            types.FsSlice `json:"labels,omitempty" yaml:"labels,omitempty"`
	TemplateLabels    types.FsSlice `json:"templateLabels,omitempty" yaml:"templateLabels,omitempty"`
	CommonAnnotations types.FsSlice `json:"commonAnnotations,omitempty" yaml:"commonAnnotations,omitempty"`
	NameReference     nbrSlice      `json:"nameReference,omitempty" yaml:"nameReference,omitempty"`
	VarReference      types.FsSlice `json:"varReference,omitempty" yaml:"varReference,omitempty"`
	Images            types.FsSlice `json:"images,omitempty" yaml:"images,omitempty"`
	Replicas          types.FsSlice `json:"replicas,omitempty" yaml:"replicas,omitempty"`
}

// MakeEmptyConfig returns an empty TransformerConfig object
func MakeEmptyConfig() *TransformerConfig {
	return &TransformerConfig{}
}

// DeepCopy returns a new copy of TransformerConfig
func (t *TransformerConfig) DeepCopy() *TransformerConfig {
	return &TransformerConfig{
		NamePrefix:        t.NamePrefix.DeepCopy(),
		NameSuffix:        t.NameSuffix.DeepCopy(),
		NameSpace:         t.NameSpace.DeepCopy(),
		CommonLabels:      t.CommonLabels.DeepCopy(),
		Labels:            t.Labels.DeepCopy(),
		TemplateLabels:    t.TemplateLabels.DeepCopy(),
		CommonAnnotations: t.CommonAnnotations.DeepCopy(),
		NameReference:     t.NameReference.DeepCopy(),
		VarReference:      t.VarReference.DeepCopy(),
		Images:            t.Images.DeepCopy(),
		Replicas:          t.Replicas.DeepCopy(),
	}
}

// the default transformer config is initialized by MakeDefaultConfig,
// and must only be accessed via that function.
var (
	initDefaultConfig sync.Once          //nolint:gochecknoglobals
	defaultConfig     *TransformerConfig //nolint:gochecknoglobals
)

// MakeDefaultConfig returns a default TransformerConfig.
func MakeDefaultConfig() *TransformerConfig {
	// parsing is expensive when having a large tree with many kustomization modules, so only do it once
	initDefaultConfig.Do(func() {
		var err error
		defaultConfig, err = makeTransformerConfigFromBytes(
			builtinpluginconsts.GetDefaultFieldSpecs())
		if err != nil {
			log.Fatalf("Unable to make default transformconfig: %v", err)
		}
	})

	// return a copy to avoid any mutations to protect the reference copy
	return defaultConfig.DeepCopy()
}

// MakeTransformerConfig returns a merger of custom config,
// if any, with default config.
func MakeTransformerConfig(
	ldr ifc.Loader, paths []string) (*TransformerConfig, error) {
	t1 := MakeDefaultConfig()
	if len(paths) == 0 {
		return t1, nil
	}
	t2, err := loadDefaultConfig(ldr, paths)
	if err != nil {
		return nil, err
	}
	return t1.Merge(t2)
}

// sortFields provides determinism in logging, tests, etc.
func (t *TransformerConfig) sortFields() {
	sort.Sort(t.NamePrefix)
	sort.Sort(t.NameSuffix)
	sort.Sort(t.NameSpace)
	sort.Sort(t.CommonLabels)
	sort.Sort(t.Labels)
	sort.Sort(t.TemplateLabels)
	sort.Sort(t.CommonAnnotations)
	sort.Sort(t.NameReference)
	sort.Sort(t.VarReference)
	sort.Sort(t.Images)
	sort.Sort(t.Replicas)
}

// AddPrefixFieldSpec adds a FieldSpec to NamePrefix
func (t *TransformerConfig) AddPrefixFieldSpec(fs types.FieldSpec) (err error) {
	t.NamePrefix, err = t.NamePrefix.MergeOne(fs)
	return err
}

// AddSuffixFieldSpec adds a FieldSpec to NameSuffix
func (t *TransformerConfig) AddSuffixFieldSpec(fs types.FieldSpec) (err error) {
	t.NameSuffix, err = t.NameSuffix.MergeOne(fs)
	return err
}

// AddCommonLabelsFieldSpec adds a FieldSpec to CommonLabels
func (t *TransformerConfig) AddCommonLabelsFieldSpec(fs types.FieldSpec) (err error) {
	t.CommonLabels, err = t.CommonLabels.MergeOne(fs)
	return err
}

// AddLabelsFieldSpec adds a FieldSpec to Labels
func (t *TransformerConfig) AddLabelsFieldSpec(fs types.FieldSpec) (err error) {
	t.Labels, err = t.Labels.MergeOne(fs)
	return err //nolint:wrapcheck
}

// AddAnnotationFieldSpec adds a FieldSpec to CommonAnnotations
func (t *TransformerConfig) AddAnnotationFieldSpec(fs types.FieldSpec) (err error) {
	t.CommonAnnotations, err = t.CommonAnnotations.MergeOne(fs)
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
	merged.NamePrefix, err = t.NamePrefix.MergeAll(input.NamePrefix)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge NamePrefix fieldSpec")
	}
	merged.NameSuffix, err = t.NameSuffix.MergeAll(input.NameSuffix)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge NameSuffix fieldSpec")
	}
	merged.NameSpace, err = t.NameSpace.MergeAll(input.NameSpace)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge NameSpace fieldSpec")
	}
	merged.CommonAnnotations, err = t.CommonAnnotations.MergeAll(
		input.CommonAnnotations)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge CommonAnnotations fieldSpec")
	}
	merged.CommonLabels, err = t.CommonLabels.MergeAll(input.CommonLabels)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge CommonLabels fieldSpec")
	}
	merged.Labels, err = t.Labels.MergeAll(input.Labels)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge Labels fieldSpec")
	}
	merged.TemplateLabels, err = t.TemplateLabels.MergeAll(input.TemplateLabels)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge TemplateLabels fieldSpec")
	}
	merged.VarReference, err = t.VarReference.MergeAll(input.VarReference)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge VarReference fieldSpec")
	}
	merged.NameReference, err = t.NameReference.mergeAll(input.NameReference)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge NameReference fieldSpec")
	}
	merged.Images, err = t.Images.MergeAll(input.Images)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge Images fieldSpec")
	}
	merged.Replicas, err = t.Replicas.MergeAll(input.Replicas)
	if err != nil {
		return nil, errors.WrapPrefixf(err, "failed to merge Replicas fieldSpec")
	}
	merged.sortFields()
	return merged, nil
}
