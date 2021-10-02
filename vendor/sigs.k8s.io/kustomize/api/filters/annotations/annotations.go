// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package annotations

import (
	"sigs.k8s.io/kustomize/api/filters/filtersutil"
	"sigs.k8s.io/kustomize/api/filters/fsslice"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type annoMap map[string]string

type Filter struct {
	// Annotations is the set of annotations to apply to the inputs
	Annotations annoMap `yaml:"annotations,omitempty"`

	// FsSlice contains the FieldSpecs to locate the namespace field
	FsSlice types.FsSlice
}

var _ kio.Filter = Filter{}

func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	keys := yaml.SortedMapKeys(f.Annotations)
	_, err := kio.FilterAll(yaml.FilterFunc(
		func(node *yaml.RNode) (*yaml.RNode, error) {
			for _, k := range keys {
				if err := node.PipeE(fsslice.Filter{
					FsSlice: f.FsSlice,
					SetValue: filtersutil.SetEntry(
						k, f.Annotations[k], yaml.NodeTagString),
					CreateKind: yaml.MappingNode, // Annotations are MappingNodes.
					CreateTag:  yaml.NodeTagMap,
				}); err != nil {
					return nil, err
				}
			}
			return node, nil
		})).Filter(nodes)
	return nodes, err
}
