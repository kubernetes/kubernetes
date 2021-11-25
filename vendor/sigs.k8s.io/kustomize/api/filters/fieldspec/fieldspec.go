// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package fieldspec

import (
	"fmt"
	"strings"

	"sigs.k8s.io/kustomize/api/filters/filtersutil"
	"sigs.k8s.io/kustomize/api/internal/utils"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/resid"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

var _ yaml.Filter = Filter{}

// Filter possibly mutates its object argument using a FieldSpec.
// If the object matches the FieldSpec, and the node found
// by following the fieldSpec's path is non-null, this filter calls
// the setValue function on the node at the end of the path.
// If any part of the path doesn't exist, the filter returns
// without doing anything and without error, unless it was set
// to create the path. If set to create, it creates a tree of maps
// along the path, and the leaf node gets the setValue called on it.
// Error on GVK mismatch, empty or poorly formed path.
// Filter expect kustomize style paths, not JSON paths.
// Filter stores internal state and should not be reused
type Filter struct {
	// FieldSpec contains the path to the value to set.
	FieldSpec types.FieldSpec `yaml:"fieldSpec"`

	// Set the field using this function
	SetValue filtersutil.SetFn

	// CreateKind defines the type of node to create if the field is not found
	CreateKind yaml.Kind

	CreateTag string

	// path keeps internal state about the current path
	path []string
}

func (fltr Filter) Filter(obj *yaml.RNode) (*yaml.RNode, error) {
	// check if the FieldSpec applies to the object
	if match := isMatchGVK(fltr.FieldSpec, obj); !match {
		return obj, nil
	}
	fltr.path = utils.PathSplitter(fltr.FieldSpec.Path, "/")
	if err := fltr.filter(obj); err != nil {
		s, _ := obj.String()
		return nil, errors.WrapPrefixf(err,
			"considering field '%s' of object\n%v", fltr.FieldSpec.Path, s)
	}
	return obj, nil
}

// Recursively called.
func (fltr Filter) filter(obj *yaml.RNode) error {
	if len(fltr.path) == 0 {
		// found the field -- set its value
		return fltr.SetValue(obj)
	}
	if obj.IsTaggedNull() || obj.IsNil() {
		return nil
	}
	switch obj.YNode().Kind {
	case yaml.SequenceNode:
		return fltr.handleSequence(obj)
	case yaml.MappingNode:
		return fltr.handleMap(obj)
	case yaml.AliasNode:
		return fltr.filter(yaml.NewRNode(obj.YNode().Alias))
	default:
		return errors.Errorf("expected sequence or mapping node")
	}
}

// handleMap calls filter on the map field matching the next path element
func (fltr Filter) handleMap(obj *yaml.RNode) error {
	fieldName, isSeq := isSequenceField(fltr.path[0])
	if fieldName == "" {
		return fmt.Errorf("cannot set or create an empty field name")
	}
	// lookup the field matching the next path element
	var operation yaml.Filter
	var kind yaml.Kind
	tag := yaml.NodeTagEmpty
	switch {
	case !fltr.FieldSpec.CreateIfNotPresent || fltr.CreateKind == 0 || isSeq:
		// don't create the field if we don't find it
		operation = yaml.Lookup(fieldName)
		if isSeq {
			// The query path thinks this field should be a sequence;
			// accept this hint for use later if the tag is NodeTagNull.
			kind = yaml.SequenceNode
		}
	case len(fltr.path) <= 1:
		// create the field if it is missing: use the provided node kind
		operation = yaml.LookupCreate(fltr.CreateKind, fieldName)
		kind = fltr.CreateKind
		tag = fltr.CreateTag
	default:
		// create the field if it is missing: must be a mapping node
		operation = yaml.LookupCreate(yaml.MappingNode, fieldName)
		kind = yaml.MappingNode
		tag = yaml.NodeTagMap
	}

	// locate (or maybe create) the field
	field, err := obj.Pipe(operation)
	if err != nil {
		return errors.WrapPrefixf(err, "fieldName: %s", fieldName)
	}
	if field == nil {
		// No error if field not found.
		return nil
	}

	// if the value exists, but is null and kind is set,
	// then change it to the creation type
	// TODO: update yaml.LookupCreate to support this
	if field.YNode().Tag == yaml.NodeTagNull && yaml.IsCreate(kind) {
		field.YNode().Kind = kind
		field.YNode().Tag = tag
	}

	// copy the current fltr and change the path on the copy
	var next = fltr
	// call filter for the next path element on the matching field
	next.path = fltr.path[1:]
	return next.filter(field)
}

// seq calls filter on all sequence elements
func (fltr Filter) handleSequence(obj *yaml.RNode) error {
	if err := obj.VisitElements(func(node *yaml.RNode) error {
		// recurse on each element -- re-allocating a Filter is
		// not strictly required, but is more consistent with field
		// and less likely to have side effects
		// keep the entire path -- it does not contain parts for sequences
		return fltr.filter(node)
	}); err != nil {
		return errors.WrapPrefixf(err,
			"visit traversal on path: %v", fltr.path)
	}
	return nil
}

// isSequenceField returns true if the path element is for a sequence field.
// isSequence also returns the path element with the '[]' suffix trimmed
func isSequenceField(name string) (string, bool) {
	shorter := strings.TrimSuffix(name, "[]")
	return shorter, shorter != name
}

// isMatchGVK returns true if the fs.GVK matches the obj GVK.
func isMatchGVK(fs types.FieldSpec, obj *yaml.RNode) bool {
	if kind := obj.GetKind(); fs.Kind != "" && fs.Kind != kind {
		// kind doesn't match
		return false
	}

	// parse the group and version from the apiVersion field
	group, version := resid.ParseGroupVersion(obj.GetApiVersion())

	if fs.Group != "" && fs.Group != group {
		// group doesn't match
		return false
	}

	if fs.Version != "" && fs.Version != version {
		// version doesn't match
		return false
	}

	return true
}
