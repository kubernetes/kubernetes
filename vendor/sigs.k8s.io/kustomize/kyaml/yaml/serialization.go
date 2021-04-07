// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package yaml

import "gopkg.in/yaml.v3"

func DoSerializationHacksOnNodes(nodes []*RNode) {
	for _, node := range nodes {
		DoSerializationHacks(node.YNode())
	}
}

// DoSerializationHacks addresses a bug in yaml V3 upstream, it parses the yaml node,
// and rearranges the head comments of the children of sequence node.
// Refer to https://github.com/go-yaml/yaml/issues/587 for more details
func DoSerializationHacks(node *yaml.Node) {
	switch node.Kind {
	case DocumentNode:
		for _, node := range node.Content {
			DoSerializationHacks(node)
		}

	case MappingNode:
		for _, node := range node.Content {
			DoSerializationHacks(node)
		}

	case SequenceNode:
		for _, node := range node.Content {
			// for each child mapping node, transfer the head comment of it's
			// first child scalar node to the head comment of itself
			// This is necessary to address serialization issue
			// https://github.com/go-yaml/yaml/issues/587 in go-yaml.v3
			// Remove this hack when the issue has been resolved
			if len(node.Content) > 0 && node.Content[0].Kind == ScalarNode {
				node.HeadComment = node.Content[0].HeadComment
				node.Content[0].HeadComment = ""
			}
		}
	}
}

func UndoSerializationHacksOnNodes(nodes []*RNode) {
	for _, node := range nodes {
		UndoSerializationHacks(node.YNode())
	}
}

// UndoSerializationHacks reverts the changes made by DoSerializationHacks
// Refer to https://github.com/go-yaml/yaml/issues/587 for more details
func UndoSerializationHacks(node *yaml.Node) {
	switch node.Kind {
	case DocumentNode:
		for _, node := range node.Content {
			DoSerializationHacks(node)
		}

	case MappingNode:
		for _, node := range node.Content {
			DoSerializationHacks(node)
		}

	case SequenceNode:
		for _, node := range node.Content {
			// revert the changes made in DoSerializationHacks
			// This is necessary to address serialization issue
			// https://github.com/go-yaml/yaml/issues/587 in go-yaml.v3
			// Remove this hack when the issue has been resolved
			if len(node.Content) > 0 && node.Content[0].Kind == ScalarNode {
				node.Content[0].HeadComment = node.HeadComment
				node.HeadComment = ""
			}
		}
	}
}
