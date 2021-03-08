// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package merge2

import (
	"fmt"

	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// A strategic merge patch directive.
// See https://github.com/kubernetes/community/blob/master/contributors/devel/sig-api-machinery/strategic-merge-patch.md
//
//go:generate stringer -type=smpDirective -linecomment
type smpDirective int

const (
	smpUnknown smpDirective = iota // unknown
	smpReplace                     // replace
	smpDelete                      // delete
	smpMerge                       // merge
)

const strategicMergePatchDirectiveKey = "$patch"

// Examine patch for a strategic merge patch directive.
// If found, return it, and remove the directive from the patch.
func determineSmpDirective(patch *yaml.RNode) (smpDirective, error) {
	if patch == nil {
		return smpMerge, nil
	}
	switch patch.YNode().Kind {
	case yaml.SequenceNode:
		return determineSequenceNodePatchStrategy(patch)
	case yaml.MappingNode:
		return determineMappingNodePatchStrategy(patch)
	default:
		return smpUnknown, fmt.Errorf(
			"no implemented strategic merge patch strategy for '%s' ('%s')",
			patch.YNode().ShortTag(), patch.MustString())
	}
}

func determineSequenceNodePatchStrategy(patch *yaml.RNode) (smpDirective, error) {
	// get the $patch element
	node, err := patch.Pipe(yaml.GetElementByKey(strategicMergePatchDirectiveKey))
	// if there are more than 1 key/value pair in the map, then this $patch
	// is not for the sequence
	if err != nil || node == nil || node.YNode() == nil || len(node.Content()) > 2 {
		return smpMerge, nil
	}
	// get the value
	value, err := node.Pipe(yaml.Get(strategicMergePatchDirectiveKey))
	if err != nil || value == nil || value.YNode() == nil {
		return smpMerge, nil
	}
	v := value.YNode().Value
	if v == smpDelete.String() {
		return smpDelete, elideSequencePatchDirective(patch, v)
	}
	if v == smpReplace.String() {
		return smpReplace, elideSequencePatchDirective(patch, v)
	}
	if v == smpMerge.String() {
		return smpMerge, elideSequencePatchDirective(patch, v)
	}
	return smpUnknown, fmt.Errorf(
		"unknown patch strategy '%s'", v)
}

func determineMappingNodePatchStrategy(patch *yaml.RNode) (smpDirective, error) {
	node, err := patch.Pipe(yaml.Get(strategicMergePatchDirectiveKey))
	if err != nil || node == nil || node.YNode() == nil {
		return smpMerge, nil
	}
	v := node.YNode().Value
	if v == smpDelete.String() {
		return smpDelete, elideMappingPatchDirective(patch)
	}
	if v == smpReplace.String() {
		return smpReplace, elideMappingPatchDirective(patch)
	}
	if v == smpMerge.String() {
		return smpMerge, elideMappingPatchDirective(patch)
	}
	return smpUnknown, fmt.Errorf(
		"unknown patch strategy '%s'", v)
}

func elideMappingPatchDirective(patch *yaml.RNode) error {
	return patch.PipeE(yaml.Clear(strategicMergePatchDirectiveKey))
}

func elideSequencePatchDirective(patch *yaml.RNode, value string) error {
	return patch.PipeE(yaml.ElementSetter{
		Element: nil,
		Keys:    []string{strategicMergePatchDirectiveKey},
		Values:  []string{value},
	})
}
