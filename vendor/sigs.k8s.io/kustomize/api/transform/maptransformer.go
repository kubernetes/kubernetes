// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transform

import (
	"errors"
	"fmt"

	"sigs.k8s.io/kustomize/api/resmap"
	"sigs.k8s.io/kustomize/api/types"
)

// mapTransformer applies a string->string map to fieldSpecs.
type mapTransformer struct {
	m          map[string]string
	fieldSpecs []types.FieldSpec
}

var _ resmap.Transformer = &mapTransformer{}

// NewMapTransformer construct a mapTransformer.
func NewMapTransformer(
	pc []types.FieldSpec, m map[string]string) (resmap.Transformer, error) {
	if m == nil {
		return newNoOpTransformer(), nil
	}
	if pc == nil {
		return nil, errors.New("fieldSpecs is not expected to be nil")
	}
	return &mapTransformer{fieldSpecs: pc, m: m}, nil
}

// Transform apply each <key, value> pair in the mapTransformer to the
// fields specified in mapTransformer.
func (o *mapTransformer) Transform(m resmap.ResMap) error {
	for _, r := range m.Resources() {
		for _, path := range o.fieldSpecs {
			if !r.OrgId().IsSelected(&path.Gvk) {
				continue
			}
			err := MutateField(
				r.Map(), path.PathSlice(),
				path.CreateIfNotPresent, o.addMap)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (o *mapTransformer) addMap(in interface{}) (interface{}, error) {
	m, ok := in.(map[string]interface{})
	if in == nil {
		m = map[string]interface{}{}
	} else if !ok {
		return nil, fmt.Errorf("%#v is expected to be %T", in, m)
	}
	for k, v := range o.m {
		m[k] = v
	}
	return m, nil
}
