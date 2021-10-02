// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package builtinconfig

import (
	"sigs.k8s.io/kustomize/api/ifc"
	"sigs.k8s.io/yaml"
)

// loadDefaultConfig returns a TranformerConfig
// object from a list of files.
func loadDefaultConfig(
	ldr ifc.Loader, paths []string) (*TransformerConfig, error) {
	result := &TransformerConfig{}
	for _, path := range paths {
		data, err := ldr.Load(path)
		if err != nil {
			return nil, err
		}
		t, err := makeTransformerConfigFromBytes(data)
		if err != nil {
			return nil, err
		}
		result, err = result.Merge(t)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// makeTransformerConfigFromBytes returns a TransformerConfig object from bytes
func makeTransformerConfigFromBytes(data []byte) (*TransformerConfig, error) {
	var t TransformerConfig
	err := yaml.Unmarshal(data, &t)
	if err != nil {
		return nil, err
	}
	t.sortFields()
	return &t, nil
}
