// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package order

import (
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// SyncOrder recursively sorts the map node keys in 'to' node to match the order of
// map node keys in 'from' node at same tree depth, additional keys are moved to the end
// Field order might be altered due to round-tripping in arbitrary functions.
// This functionality helps to retain the original order of fields to avoid unnecessary diffs.
func SyncOrder(from, to *yaml.RNode) error {
	// from node should not be modified, it should be just used as a reference
	fromCopy := from.Copy()
	if err := syncOrder(fromCopy, to); err != nil {
		return errors.Errorf("failed to sync field order: %q", err.Error())
	}
	rearrangeHeadCommentOfSeqNode(to.YNode())
	return nil
}

func syncOrder(from, to *yaml.RNode) error {
	if from.IsNilOrEmpty() || to.IsNilOrEmpty() {
		return nil
	}
	switch from.YNode().Kind {
	case yaml.DocumentNode:
		// Traverse the child of the documents
		return syncOrder(yaml.NewRNode(from.YNode()), yaml.NewRNode(to.YNode()))
	case yaml.MappingNode:
		return VisitFields(from, to, func(fNode, tNode *yaml.MapNode) error {
			// Traverse each field value
			if fNode == nil || tNode == nil {
				return nil
			}
			return syncOrder(fNode.Value, tNode.Value)
		})
	case yaml.SequenceNode:
		return VisitElements(from, to, func(fNode, tNode *yaml.RNode) error {
			// Traverse each list element
			return syncOrder(fNode, tNode)
		})
	}
	return nil
}

// VisitElements calls fn for each element in a SequenceNode.
// Returns an error for non-SequenceNodes
func VisitElements(from, to *yaml.RNode, fn func(fNode, tNode *yaml.RNode) error) error {
	fElements, err := from.Elements()
	if err != nil {
		return errors.Wrap(err)
	}

	tElements, err := to.Elements()
	if err != nil {
		return errors.Wrap(err)
	}
	for i := range fElements {
		if i >= len(tElements) {
			return nil
		}
		if err := fn(fElements[i], tElements[i]); err != nil {
			return errors.Wrap(err)
		}
	}
	return nil
}

// VisitFields calls fn for each field in the RNode.
// Returns an error for non-MappingNodes.
func VisitFields(from, to *yaml.RNode, fn func(fNode, tNode *yaml.MapNode) error) error {
	srcFieldNames, err := from.Fields()
	if err != nil {
		return nil
	}
	yaml.SyncMapNodesOrder(from, to)
	// visit each field
	for _, fieldName := range srcFieldNames {
		if err := fn(from.Field(fieldName), to.Field(fieldName)); err != nil {
			return errors.Wrap(err)
		}
	}
	return nil
}

// rearrangeHeadCommentOfSeqNode addresses a remote corner case due to moving a
// map node in a sequence node with a head comment to the top
func rearrangeHeadCommentOfSeqNode(node *yaml.Node) {
	if node == nil {
		return
	}
	switch node.Kind {
	case yaml.DocumentNode:
		for _, node := range node.Content {
			rearrangeHeadCommentOfSeqNode(node)
		}

	case yaml.MappingNode:
		for _, node := range node.Content {
			rearrangeHeadCommentOfSeqNode(node)
		}

	case yaml.SequenceNode:
		for _, node := range node.Content {
			// for each child mapping node, transfer the head comment of it's
			// first child scalar node to the head comment of itself
			if len(node.Content) > 0 && node.Content[0].Kind == yaml.ScalarNode {
				if node.HeadComment == "" {
					node.HeadComment = node.Content[0].HeadComment
					continue
				}

				if node.Content[0].HeadComment != "" {
					node.HeadComment += "\n" + node.Content[0].HeadComment
					node.Content[0].HeadComment = ""
				}
			}
		}
	}
}
