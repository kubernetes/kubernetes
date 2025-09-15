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
	// from node should not be modified, it should be just used as a reference
	fromCopy := from.Copy()
	copyFieldComments(fromCopy, to)
	// walk the fields copying comments
	_, err := walk.Walker{
		Sources:            []*yaml.RNode{fromCopy, to},
		Visitor:            &copier{},
		VisitKeysAsScalars: true}.Walk()
	return err
}

// copier implements walk.Visitor, and copies comments to fields shared between 2 instances
// of a resource
type copier struct{}

func (c *copier) VisitMap(s walk.Sources, _ *openapi.ResourceSchema) (*yaml.RNode, error) {
	copyFieldComments(s.Dest(), s.Origin())
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

	copyFieldComments(s.Dest(), to)
	return s.Dest(), nil
}

func (c *copier) VisitList(s walk.Sources, _ *openapi.ResourceSchema, _ walk.ListKind) (
	*yaml.RNode, error) {
	copyFieldComments(s.Dest(), s.Origin())
	destItems := s.Dest().Content()
	originItems := s.Origin().Content()

	for i := 0; i < len(destItems) && i < len(originItems); i++ {
		dest := destItems[i]
		origin := originItems[i]

		if dest.Value == origin.Value {
			// We copy the comments recursively on each node in the list.
			if err := CopyComments(yaml.NewRNode(dest), yaml.NewRNode(origin)); err != nil {
				return nil, err
			}
		}
	}

	return s.Dest(), nil
}

// copyFieldComments copies the comment from one field to another
func copyFieldComments(from, to *yaml.RNode) {
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
