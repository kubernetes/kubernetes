// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"fmt"

	"sigs.k8s.io/kustomize/api/resid"
)

// FieldSpec completely specifies a kustomizable field in a k8s API object.
// It helps define the operands of transformations.
//
// For example, a directive to add a common label to objects
// will need to know that a 'Deployment' object (in API group
// 'apps', any version) can have labels at field path
// 'spec/template/metadata/labels', and further that it is OK
// (or not OK) to add that field path to the object if the
// field path doesn't exist already.
//
// This would look like
// {
//   group: apps
//   kind: Deployment
//   path: spec/template/metadata/labels
//   create: true
// }
type FieldSpec struct {
	resid.Gvk          `json:",inline,omitempty" yaml:",inline,omitempty"`
	Path               string `json:"path,omitempty" yaml:"path,omitempty"`
	CreateIfNotPresent bool   `json:"create,omitempty" yaml:"create,omitempty"`
}

func (fs FieldSpec) String() string {
	return fmt.Sprintf(
		"%s:%v:%s", fs.Gvk.String(), fs.CreateIfNotPresent, fs.Path)
}

// If true, the primary key is the same, but other fields might not be.
func (fs FieldSpec) effectivelyEquals(other FieldSpec) bool {
	return fs.IsSelected(&other.Gvk) && fs.Path == other.Path
}

type FsSlice []FieldSpec

func (s FsSlice) Len() int      { return len(s) }
func (s FsSlice) Swap(i, j int) { s[i], s[j] = s[j], s[i] }
func (s FsSlice) Less(i, j int) bool {
	return s[i].Gvk.IsLessThan(s[j].Gvk)
}

// MergeAll merges the argument into this, returning the result.
// Items already present are ignored.
// Items that conflict (primary key matches, but remain data differs)
// result in an error.
func (s FsSlice) MergeAll(incoming FsSlice) (result FsSlice, err error) {
	result = s
	for _, x := range incoming {
		result, err = result.MergeOne(x)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// MergeOne merges the argument into this, returning the result.
// If the item's primary key is already present, and there are no
// conflicts, it is ignored (we don't want duplicates).
// If there is a conflict, the merge fails.
func (s FsSlice) MergeOne(x FieldSpec) (FsSlice, error) {
	i := s.index(x)
	if i > -1 {
		// It's already there.
		if s[i].CreateIfNotPresent != x.CreateIfNotPresent {
			return nil, fmt.Errorf("conflicting fieldspecs")
		}
		return s, nil
	}
	return append(s, x), nil
}

func (s FsSlice) index(fs FieldSpec) int {
	for i, x := range s {
		if x.effectivelyEquals(fs) {
			return i
		}
	}
	return -1
}
