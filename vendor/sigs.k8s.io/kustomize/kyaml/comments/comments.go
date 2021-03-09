// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package comments

import (
	"sigs.k8s.io/kustomize/kyaml/openapi"
	"sigs.k8s.io/kustomize/kyaml/yaml"
	"sigs.k8s.io/kustomize/kyaml/yaml/walk"
)

// CopyComments recursively copies the comments on fields in from to fields in to
func CopyComments(from, to *yaml.RNode) error {
	copy(from, to)
	// walk the fields copying comments
	_, err := walk.Walker{
		Sources:            []*yaml.RNode{from, to},
		Visitor:            &copier{},
		VisitKeysAsScalars: true}.Walk()
	return err
}

// copier implements walk.Visitor, and copies comments to fields shared between 2 instances
// of a resource
type copier struct{}

func (c *copier) VisitMap(s walk.Sources, _ *openapi.ResourceSchema) (*yaml.RNode, error) {
	copy(s.Dest(), s.Origin())
	return s.Dest(), nil
}

func (c *copier) VisitScalar(s walk.Sources, _ *openapi.ResourceSchema) (*yaml.RNode, error) {
	to := s.Origin()
	// TODO: File a bug with upstream yaml to handle comments for FoldedStyle scalar nodes
	// Hack: convert FoldedStyle scalar node to DoubleQuotedStyle as the line comments are
	// being serialized without space
	// https://github.com/GoogleContainerTools/kpt/issues/766
	if to != nil && to.Document().Style == yaml.FoldedStyle {
		to.Document().Style = yaml.DoubleQuotedStyle
	}

	copy(s.Dest(), to)
	return s.Dest(), nil
}

func (c *copier) VisitList(s walk.Sources, _ *openapi.ResourceSchema, _ walk.ListKind) (
	*yaml.RNode, error) {
	copy(s.Dest(), s.Origin())
	destItems := s.Dest().Content()
	originItems := s.Origin().Content()

	for i := 0; i < len(destItems) && i < len(originItems); i++ {
		dest := destItems[i]
		origin := originItems[i]

		if dest.Value == origin.Value {
			copy(yaml.NewRNode(dest), yaml.NewRNode(origin))
		}
	}

	return s.Dest(), nil
}

// copy copies the comment from one field to another
func copy(from, to *yaml.RNode) {
	if from == nil || to == nil {
		return
	}
	if to.Document().LineComment == "" {
		to.Document().LineComment = from.Document().LineComment
	}
	if to.Document().HeadComment == "" {
		to.Document().HeadComment = from.Document().HeadComment
	}
	if to.Document().FootComment == "" {
		to.Document().FootComment = from.Document().FootComment
	}
}
