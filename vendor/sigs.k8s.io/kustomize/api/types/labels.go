// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

type Label struct {
	// Pairs contains the key-value pairs for labels to add
	Pairs map[string]string `json:"pairs,omitempty" yaml:"pairs,omitempty"`
	// IncludeSelectors inidicates should transformer include the
	// fieldSpecs for selectors. Custom fieldSpecs specified by
	// FieldSpecs will be merged with builtin fieldSpecs if this
	// is true.
	IncludeSelectors bool        `json:"includeSelectors,omitempty" yaml:"includeSelectors,omitempty"`
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
