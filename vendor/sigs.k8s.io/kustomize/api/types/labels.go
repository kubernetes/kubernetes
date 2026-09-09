// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

type Label struct {
	// Pairs contains the key-value pairs for labels to add
	Pairs map[string]string `json:"pairs,omitempty" yaml:"pairs,omitempty"`
	// IncludeSelectors indicates whether the transformer should include the
	// fieldSpecs for selectors. Custom fieldSpecs specified by
	// FieldSpecs will be merged with builtin fieldSpecs if this
	// is true.
	IncludeSelectors bool `json:"includeSelectors,omitempty" yaml:"includeSelectors,omitempty"`
	// IncludeTemplates indicates whether the transformer should include the
	// spec/template/metadata fieldSpec. Custom fieldSpecs specified by
	// FieldSpecs will be merged with spec/template/metadata fieldSpec if this
	// is true. If IncludeSelectors is true, IncludeTemplates is not needed.
	IncludeTemplates bool        `json:"includeTemplates,omitempty" yaml:"includeTemplates,omitempty"`
	FieldSpecs       []FieldSpec `json:"fields,omitempty" yaml:"fields,omitempty"`
}

func labelFromCommonLabels(commonLabels map[string]string) *Label {
	if len(commonLabels) == 0 {
		return nil
	}
	return &Label{
		Pairs:            commonLabels,
		IncludeSelectors: true,
	}
}
