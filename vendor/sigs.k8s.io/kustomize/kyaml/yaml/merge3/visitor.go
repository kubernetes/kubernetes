// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package merge3

import (
	"sigs.k8s.io/kustomize/kyaml/openapi"
	"sigs.k8s.io/kustomize/kyaml/yaml"
	"sigs.k8s.io/kustomize/kyaml/yaml/walk"
)

type ConflictStrategy uint

const (
	// TODO: Support more strategies
	TakeUpdate ConflictStrategy = 1 + iota
)

type Visitor struct{}

func (m Visitor) VisitMap(nodes walk.Sources, s *openapi.ResourceSchema) (*yaml.RNode, error) {
	if nodes.Updated().IsTaggedNull() || nodes.Dest().IsTaggedNull() {
		// explicitly cleared from either dest or update
		return walk.ClearNode, nil
	}
	if nodes.Dest() == nil && nodes.Updated() == nil {
		// implicitly cleared missing from both dest and update
		return walk.ClearNode, nil
	}

	if nodes.Dest() == nil {
		// not cleared, but missing from the dest
		// initialize a new value that can be recursively merged
		return yaml.NewRNode(&yaml.Node{Kind: yaml.MappingNode}), nil
	}

	// recursively merge the dest with the original and updated
	return nodes.Dest(), nil
}

func (m Visitor) visitAList(nodes walk.Sources, _ *openapi.ResourceSchema) (*yaml.RNode, error) {
	if yaml.IsMissingOrNull(nodes.Updated()) && !yaml.IsMissingOrNull(nodes.Origin()) {
		// implicitly cleared from update -- element was deleted
		return walk.ClearNode, nil
	}
	if yaml.IsMissingOrNull(nodes.Dest()) {
		// not cleared, but missing from the dest
		// initialize a new value that can be recursively merged
		return yaml.NewRNode(&yaml.Node{Kind: yaml.SequenceNode}), nil
	}

	// recursively merge the dest with the original and updated
	return nodes.Dest(), nil
}

func (m Visitor) VisitScalar(nodes walk.Sources, s *openapi.ResourceSchema) (*yaml.RNode, error) {
	if nodes.Updated().IsTaggedNull() || nodes.Dest().IsTaggedNull() {
		// explicitly cleared from either dest or update
		return nil, nil
	}
	if yaml.IsMissingOrNull(nodes.Updated()) != yaml.IsMissingOrNull(nodes.Origin()) {
		// value added or removed in update
		return nodes.Updated(), nil
	}
	if yaml.IsMissingOrNull(nodes.Updated()) && yaml.IsMissingOrNull(nodes.Origin()) {
		// value added or removed in update
		return nodes.Dest(), nil
	}

	values, err := m.getStrValues(nodes)
	if err != nil {
		return nil, err
	}

	if (values.Dest == "" || values.Dest == values.Origin) && values.Origin != values.Update {
		// if local is nil or is unchanged but there is new update
		return nodes.Updated(), nil
	}

	if nodes.Updated().YNode().Value != nodes.Origin().YNode().Value {
		// value changed in update
		return nodes.Updated(), nil
	}

	// unchanged between origin and update, keep the dest
	return nodes.Dest(), nil
}

func (m Visitor) visitNAList(nodes walk.Sources) (*yaml.RNode, error) {
	if nodes.Updated().IsTaggedNull() || nodes.Dest().IsTaggedNull() {
		// explicitly cleared from either dest or update
		return walk.ClearNode, nil
	}

	if yaml.IsMissingOrNull(nodes.Updated()) != yaml.IsMissingOrNull(nodes.Origin()) {
		// value added or removed in update
		return nodes.Updated(), nil
	}
	if yaml.IsMissingOrNull(nodes.Updated()) && yaml.IsMissingOrNull(nodes.Origin()) {
		// value not present in source or dest
		return nodes.Dest(), nil
	}

	// compare origin and update values to see if they have changed
	values, err := m.getStrValues(nodes)
	if err != nil {
		return nil, err
	}
	if values.Update != values.Origin {
		// value changed in update
		return nodes.Updated(), nil
	}

	// unchanged between origin and update, keep the dest
	return nodes.Dest(), nil
}

func (m Visitor) VisitList(nodes walk.Sources, s *openapi.ResourceSchema, kind walk.ListKind) (*yaml.RNode, error) {
	if kind == walk.AssociativeList {
		return m.visitAList(nodes, s)
	}
	// non-associative list
	return m.visitNAList(nodes)
}

func (m Visitor) getStrValues(nodes walk.Sources) (strValues, error) {
	var uStr, oStr, dStr string
	var err error
	if nodes.Updated() != nil && nodes.Updated().YNode() != nil {
		s := nodes.Updated().YNode().Style
		defer func() {
			nodes.Updated().YNode().Style = s
		}()
		nodes.Updated().YNode().Style = yaml.FlowStyle | yaml.SingleQuotedStyle
		uStr, err = nodes.Updated().String()
		if err != nil {
			return strValues{}, err
		}
	}
	if nodes.Origin() != nil && nodes.Origin().YNode() != nil {
		s := nodes.Origin().YNode().Style
		defer func() {
			nodes.Origin().YNode().Style = s
		}()
		nodes.Origin().YNode().Style = yaml.FlowStyle | yaml.SingleQuotedStyle
		oStr, err = nodes.Origin().String()
		if err != nil {
			return strValues{}, err
		}
	}
	if nodes.Dest() != nil && nodes.Dest().YNode() != nil {
		s := nodes.Dest().YNode().Style
		defer func() {
			nodes.Dest().YNode().Style = s
		}()
		nodes.Dest().YNode().Style = yaml.FlowStyle | yaml.SingleQuotedStyle
		dStr, err = nodes.Dest().String()
		if err != nil {
			return strValues{}, err
		}
	}

	return strValues{Origin: oStr, Update: uStr, Dest: dStr}, nil
}

type strValues struct {
	Origin string
	Update string
	Dest   string
}

var _ walk.Visitor = Visitor{}
