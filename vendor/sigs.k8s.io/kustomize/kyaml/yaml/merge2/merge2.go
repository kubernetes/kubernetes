// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package merge contains libraries for merging fields from one RNode to another
// RNode
package merge2

import (
	"sigs.k8s.io/kustomize/kyaml/openapi"
	"sigs.k8s.io/kustomize/kyaml/yaml"
	"sigs.k8s.io/kustomize/kyaml/yaml/walk"
)

// Merge merges fields from src into dest.
func Merge(src, dest *yaml.RNode, mergeOptions yaml.MergeOptions) (*yaml.RNode, error) {
	return walk.Walker{
		Sources:      []*yaml.RNode{dest, src},
		Visitor:      Merger{},
		MergeOptions: mergeOptions,
	}.Walk()
}

// Merge parses the arguments, and merges fields from srcStr into destStr.
func MergeStrings(srcStr, destStr string, infer bool, mergeOptions yaml.MergeOptions) (string, error) {
	src, err := yaml.Parse(srcStr)
	if err != nil {
		return "", err
	}
	dest, err := yaml.Parse(destStr)
	if err != nil {
		return "", err
	}

	result, err := walk.Walker{
		Sources:               []*yaml.RNode{dest, src},
		Visitor:               Merger{},
		InferAssociativeLists: infer,
		MergeOptions:          mergeOptions,
	}.Walk()
	if err != nil {
		return "", err
	}

	return result.String()
}

type Merger struct {
	// for forwards compatibility when new functions are added to the interface
}

var _ walk.Visitor = Merger{}

func (m Merger) VisitMap(nodes walk.Sources, s *openapi.ResourceSchema) (*yaml.RNode, error) {
	if err := m.SetComments(nodes); err != nil {
		return nil, err
	}
	if err := m.SetStyle(nodes); err != nil {
		return nil, err
	}
	if yaml.IsMissingOrNull(nodes.Dest()) {
		// Add
		ps, _ := determineSmpDirective(nodes.Origin())
		if ps == smpDelete {
			return walk.ClearNode, nil
		}

		return nodes.Origin(), nil
	}
	if nodes.Origin().IsTaggedNull() {
		// clear the value
		return walk.ClearNode, nil
	}

	ps, err := determineSmpDirective(nodes.Origin())
	if err != nil {
		return nil, err
	}

	switch ps {
	case smpDelete:
		return walk.ClearNode, nil
	case smpReplace:
		return nodes.Origin(), nil
	default:
		return nodes.Dest(), nil
	}
}

func (m Merger) VisitScalar(nodes walk.Sources, s *openapi.ResourceSchema) (*yaml.RNode, error) {
	if err := m.SetComments(nodes); err != nil {
		return nil, err
	}
	if err := m.SetStyle(nodes); err != nil {
		return nil, err
	}
	// Override value
	if nodes.Origin() != nil {
		return nodes.Origin(), nil
	}
	// Keep
	return nodes.Dest(), nil
}

func (m Merger) VisitList(nodes walk.Sources, s *openapi.ResourceSchema, kind walk.ListKind) (*yaml.RNode, error) {
	if err := m.SetComments(nodes); err != nil {
		return nil, err
	}
	if err := m.SetStyle(nodes); err != nil {
		return nil, err
	}
	if kind == walk.NonAssociateList {
		// Override value
		if nodes.Origin() != nil {
			return nodes.Origin(), nil
		}
		// Keep
		return nodes.Dest(), nil
	}

	// Add
	if yaml.IsMissingOrNull(nodes.Dest()) {
		return nodes.Origin(), nil
	}
	// Clear
	if nodes.Origin().IsTaggedNull() {
		return walk.ClearNode, nil
	}

	ps, err := determineSmpDirective(nodes.Origin())
	if err != nil {
		return nil, err
	}

	switch ps {
	case smpDelete:
		return walk.ClearNode, nil
	case smpReplace:
		return nodes.Origin(), nil
	default:
		return nodes.Dest(), nil
	}
}

func (m Merger) SetStyle(sources walk.Sources) error {
	source := sources.Origin()
	dest := sources.Dest()
	if dest == nil || dest.YNode() == nil || source == nil || source.YNode() == nil {
		// avoid panic
		return nil
	}

	// copy the style from the source.
	// special case: if the dest was an empty map or seq, then it probably had
	// folded style applied, but we actually want to keep the style of the origin
	// in this case (even if it was the default).  otherwise the merged elements
	// will get folded even though this probably isn't what is desired.
	if dest.YNode().Kind != yaml.ScalarNode && len(dest.YNode().Content) == 0 {
		dest.YNode().Style = source.YNode().Style
	}
	return nil
}

// SetComments copies the dest comments to the source comments if they are present
// on the source.
func (m Merger) SetComments(sources walk.Sources) error {
	source := sources.Origin()
	dest := sources.Dest()
	if dest == nil || dest.YNode() == nil || source == nil || source.YNode() == nil {
		// avoid panic
		return nil
	}
	if source.YNode().FootComment != "" {
		dest.YNode().FootComment = source.YNode().FootComment
	}
	if source.YNode().HeadComment != "" {
		dest.YNode().HeadComment = source.YNode().HeadComment
	}
	if source.YNode().LineComment != "" {
		dest.YNode().LineComment = source.YNode().LineComment
	}
	return nil
}
