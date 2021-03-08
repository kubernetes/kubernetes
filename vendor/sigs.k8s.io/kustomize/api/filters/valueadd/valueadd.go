// Copyright 2020 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package valueadd

import (
	"strings"

	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// An 'Add' operation aspiring to IETF RFC 6902 JSON.
//
// The filter tries to add a value to a node at a particular field path.
//
// Kinds of target fields:
//
// - Non-existent target field.
//
//   The field will be added and the value inserted.
//
// - Existing field, scalar or map.
//
//   E.g. 'spec/template/spec/containers/[name:nginx]/image'
//
//   This behaves like an IETF RFC 6902 Replace operation would;
//   the existing value is replaced without complaint, even though
//   this  is an Add operation.  In contrast, a Replace operation
//   must fail (report an error) if the field doesn't exist.
//
// - Existing field, list (array)
//   Not supported yet.
//   TODO: Honor fields with RFC-6902-style array indices
//   TODO: like 'spec/template/spec/containers/2'
//   TODO: Modify kyaml/yaml/PathGetter to allow this.
//   The value will be inserted into the array at the given position,
//   shifting other contents. To instead replace an array entry, use
//   an implementation of an IETF RFC 6902 Replace operation.
//
// For the common case of a filepath in the field value, and a desire
// to add the value to the filepath (rather than replace the filepath),
// use a non-zero value of FilePathPosition (see below).
type Filter struct {
	// Value is the value to add.
	//
	// Empty values are disallowed, i.e. this filter isn't intended
	// for use in erasing or removing fields. For that, use a filter
	// more aligned with the IETF RFC 6902 JSON Remove operation.
	//
	// At the time of writing, Value's value should be a simple string,
	// not a JSON document.  This particular filter focuses on easing
	// injection of a single-sourced cloud project and/or cluster name
	// into various fields, especially namespace and various filepath
	// specifications.
	Value string

	// FieldPath is a JSON-style path to the field intended to hold the value.
	FieldPath string

	// FilePathPosition is a filepath field index.
	//
	// Call the value of this field _i_.
	//
	//   If _i_ is zero, negative or unspecified, this field has no effect.
	//
	//   If _i_ is > 0, then it's assumed that
	//   - 'Value' is a string that can work as a directory or file name,
	//   - the field value intended for replacement holds a filepath.
	//
	// The filepath is split into a string slice, the value is inserted
	// at position [i-1], shifting the rest of the path to the right.
	// A value of i==1 puts the new value at the start of the path.
	// This change never converts an absolute path to a relative path,
	// meaning adding a new field at position i==1 will preserve a
	// leading slash. E.g. if Value == 'PEACH'
	//
	//                  OLD : NEW                    : FilePathPosition
	//      --------------------------------------------------------
	//              {empty} : PEACH                  : irrelevant
	//                    / : /PEACH                 : irrelevant
	//                  pie : PEACH/pie              : 1 (or less to prefix)
	//                 /pie : /PEACH/pie             : 1 (or less to prefix)
	//                  raw : raw/PEACH              : 2 (or more to postfix)
	//                 /raw : /raw/PEACH             : 2 (or more to postfix)
	//      a/nice/warm/pie : a/nice/warm/PEACH/pie  : 4
	//     /a/nice/warm/pie : /a/nice/warm/PEACH/pie : 4
	//
	// For robustness (liberal input, conservative output) FilePathPosition
	// values that that are too large to index the split filepath result in a
	// postfix rather than an error.  So use 1 to prefix, 9999 to postfix.
	FilePathPosition int `json:"filePathPosition,omitempty" yaml:"filePathPosition,omitempty"`
}

var _ kio.Filter = Filter{}

func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	_, err := kio.FilterAll(yaml.FilterFunc(
		func(node *yaml.RNode) (*yaml.RNode, error) {
			var fields []string
			// if there is forward slash '/' in the field name, a back slash '\'
			// will be used to escape it.
			for _, f := range strings.Split(f.FieldPath, "/") {
				if len(fields) > 0 && strings.HasSuffix(fields[len(fields)-1], "\\") {
					concatField := strings.TrimSuffix(fields[len(fields)-1], "\\") + "/" + f
					fields = append(fields[:len(fields)-1], concatField)
				} else {
					fields = append(fields, f)
				}
			}
			// TODO: support SequenceNode.
			// Presumably here one could look for array indices (digits) at
			// the end of the field path (as described in IETF RFC 6902 JSON),
			// and if found, take it as a signal that this should be a
			// SequenceNode instead of a ScalarNode, and insert the value
			// into the proper slot, shifting every over.
			n, err := node.Pipe(yaml.LookupCreate(yaml.ScalarNode, fields...))
			if err != nil {
				return node, err
			}
			// TODO: allow more kinds
			if err := yaml.ErrorIfInvalid(n, yaml.ScalarNode); err != nil {
				return nil, err
			}
			newValue := f.Value
			if f.FilePathPosition > 0 {
				newValue = filesys.InsertPathPart(
					n.YNode().Value, f.FilePathPosition-1, newValue)
			}
			return n.Pipe(yaml.FieldSetter{StringValue: newValue})
		})).Filter(nodes)
	return nodes, err
}
