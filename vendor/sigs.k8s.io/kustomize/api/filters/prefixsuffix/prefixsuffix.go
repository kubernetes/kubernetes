// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package prefixsuffix

import (
	"fmt"

	"sigs.k8s.io/kustomize/api/filters/fieldspec"
	"sigs.k8s.io/kustomize/api/filters/filtersutil"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Filter applies resource name prefix's and suffix's using the fieldSpecs
type Filter struct {
	Prefix string `json:"prefix,omitempty" yaml:"prefix,omitempty"`
	Suffix string `json:"suffix,omitempty" yaml:"suffix,omitempty"`

	FieldSpec types.FieldSpec `json:"fieldSpec,omitempty" yaml:"fieldSpec,omitempty"`
}

var _ kio.Filter = Filter{}

func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	return kio.FilterAll(yaml.FilterFunc(f.run)).Filter(nodes)
}

func (f Filter) run(node *yaml.RNode) (*yaml.RNode, error) {
	err := node.PipeE(fieldspec.Filter{
		FieldSpec:  f.FieldSpec,
		SetValue:   f.evaluateField,
		CreateKind: yaml.ScalarNode, // Name is a ScalarNode
		CreateTag:  yaml.NodeTagString,
	})
	return node, err
}

func (f Filter) evaluateField(node *yaml.RNode) error {
	return filtersutil.SetScalar(fmt.Sprintf(
		"%s%s%s", f.Prefix, node.YNode().Value, f.Suffix))(node)
}
