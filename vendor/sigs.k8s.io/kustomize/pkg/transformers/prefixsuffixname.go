/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package transformers

import (
	"errors"
	"fmt"

	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

// namePrefixSuffixTransformer contains the prefix, suffix, and the FieldSpecs
// for each field needing a name prefix and suffix.
type namePrefixSuffixTransformer struct {
	prefix           string
	suffix           string
	fieldSpecsToUse  []config.FieldSpec
	fieldSpecsToSkip []config.FieldSpec
}

var _ Transformer = &namePrefixSuffixTransformer{}

var prefixSuffixFieldSpecsToSkip = []config.FieldSpec{
	{
		Gvk: gvk.Gvk{Kind: "CustomResourceDefinition"},
	},
}

// NewNamePrefixSuffixTransformer makes a namePrefixSuffixTransformer.
func NewNamePrefixSuffixTransformer(
	np, ns string, fieldSpecs []config.FieldSpec) (Transformer, error) {
	if len(np) == 0 && len(ns) == 0 {
		return NewNoOpTransformer(), nil
	}
	if fieldSpecs == nil {
		return nil, errors.New("fieldSpecs is not expected to be nil")
	}
	return &namePrefixSuffixTransformer{
		prefix:           np,
		suffix:           ns,
		fieldSpecsToUse:  fieldSpecs,
		fieldSpecsToSkip: prefixSuffixFieldSpecsToSkip}, nil
}

// Transform prepends the name prefix and appends the name suffix.
func (o *namePrefixSuffixTransformer) Transform(m resmap.ResMap) error {
	// Fill map "mf" with entries subject to name modification, and
	// delete these entries from "m", so that for now m retains only
	// the entries whose names will not be modified.
	mf := resmap.ResMap{}
	for id := range m {
		found := false
		for _, path := range o.fieldSpecsToSkip {
			if id.Gvk().IsSelected(&path.Gvk) {
				found = true
				break
			}
		}
		if !found {
			mf[id] = m[id]
			delete(m, id)
		}
	}

	for id := range mf {
		objMap := mf[id].Map()
		for _, path := range o.fieldSpecsToUse {
			if !id.Gvk().IsSelected(&path.Gvk) {
				continue
			}
			err := mutateField(
				objMap,
				path.PathSlice(),
				path.CreateIfNotPresent,
				o.addPrefixSuffix)
			if err != nil {
				return err
			}
			newId := id.CopyWithNewPrefixSuffix(o.prefix, o.suffix)
			m[newId] = mf[id]
		}
	}
	return nil
}

func (o *namePrefixSuffixTransformer) addPrefixSuffix(
	in interface{}) (interface{}, error) {
	s, ok := in.(string)
	if !ok {
		return nil, fmt.Errorf("%#v is expected to be %T", in, s)
	}
	return fmt.Sprintf("%s%s%s", o.prefix, s, o.suffix), nil
}
