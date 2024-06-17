// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package fsslice

import (
	"sigs.k8s.io/kustomize/api/filters/fieldspec"
	"sigs.k8s.io/kustomize/api/filters/filtersutil"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

var _ yaml.Filter = Filter{}

// Filter ranges over an FsSlice to modify fields on a single object.
// An FsSlice is a range of FieldSpecs. A FieldSpec is a GVK plus a path.
type Filter struct {
	// FieldSpecList list of FieldSpecs to set
	FsSlice types.FsSlice `yaml:"fsSlice"`

	// SetValue is called on each field that matches one of the FieldSpecs
	SetValue filtersutil.SetFn

	// CreateKind is used to create fields that do not exist
	CreateKind yaml.Kind

	// CreateTag is used to set the tag if encountering a null field
	CreateTag string
}

func (fltr Filter) Filter(obj *yaml.RNode) (*yaml.RNode, error) {
	for i := range fltr.FsSlice {
		// apply this FieldSpec
		// create a new filter for each iteration because they
		// store internal state about the field paths
		_, err := (&fieldspec.Filter{
			FieldSpec:  fltr.FsSlice[i],
			SetValue:   fltr.SetValue,
			CreateKind: fltr.CreateKind,
			CreateTag:  fltr.CreateTag,
		}).Filter(obj)
		if err != nil {
			return nil, err
		}
	}
	return obj, nil
}
