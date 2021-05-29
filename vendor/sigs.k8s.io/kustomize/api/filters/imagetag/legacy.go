// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package imagetag

import (
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// LegacyFilter is an implementation of the kio.Filter interface
// that scans through the provided kyaml data structure and updates
// any values of any image fields that is inside a sequence under
// a field called either containers or initContainers. The field is only
// update if it has a value that matches and image reference and the name
// of the image is a match with the provided ImageTag.
type LegacyFilter struct {
	ImageTag types.Image `json:"imageTag,omitempty" yaml:"imageTag,omitempty"`
}

var _ kio.Filter = LegacyFilter{}

func (lf LegacyFilter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	return kio.FilterAll(yaml.FilterFunc(lf.filter)).Filter(nodes)
}

func (lf LegacyFilter) filter(node *yaml.RNode) (*yaml.RNode, error) {
	meta, err := node.GetMeta()
	if err != nil {
		return nil, err
	}

	// We do not make any changes if the type of the resource
	// is CustomResourceDefinition.
	if meta.Kind == `CustomResourceDefinition` {
		return node, nil
	}

	fff := findFieldsFilter{
		fields:        []string{"containers", "initContainers"},
		fieldCallback: checkImageTagsFn(lf.ImageTag),
	}
	if err := node.PipeE(fff); err != nil {
		return nil, err
	}
	return node, nil
}

type fieldCallback func(node *yaml.RNode) error

// findFieldsFilter is an implementation of the kio.Filter
// interface. It will walk the data structure and look for fields
// that matches the provided list of field names. For each match,
// the value of the field will be passed in as a parameter to the
// provided fieldCallback.
// TODO: move this to kyaml/filterutils
type findFieldsFilter struct {
	fields []string

	fieldCallback fieldCallback
}

func (f findFieldsFilter) Filter(obj *yaml.RNode) (*yaml.RNode, error) {
	return obj, f.walk(obj)
}

func (f findFieldsFilter) walk(node *yaml.RNode) error {
	switch node.YNode().Kind {
	case yaml.MappingNode:
		return node.VisitFields(func(n *yaml.MapNode) error {
			err := f.walk(n.Value)
			if err != nil {
				return err
			}
			key := n.Key.YNode().Value
			if contains(f.fields, key) {
				return f.fieldCallback(n.Value)
			}
			return nil
		})
	case yaml.SequenceNode:
		return node.VisitElements(func(n *yaml.RNode) error {
			return f.walk(n)
		})
	}
	return nil
}

func contains(slice []string, str string) bool {
	for _, s := range slice {
		if s == str {
			return true
		}
	}
	return false
}

func checkImageTagsFn(imageTag types.Image) fieldCallback {
	return func(node *yaml.RNode) error {
		if node.YNode().Kind != yaml.SequenceNode {
			return nil
		}

		return node.VisitElements(func(n *yaml.RNode) error {
			// Look up any fields on the provided node that is named
			// image.
			return n.PipeE(yaml.Get("image"), imageTagUpdater{
				ImageTag: imageTag,
			})
		})
	}
}
