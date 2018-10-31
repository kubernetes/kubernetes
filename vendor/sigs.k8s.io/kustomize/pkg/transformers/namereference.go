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
	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type nameReferenceTransformer struct {
	backRefs []config.NameBackReferences
}

var _ Transformer = &nameReferenceTransformer{}

// NewNameReferenceTransformer constructs a nameReferenceTransformer
// with a given slice of NameBackReferences.
func NewNameReferenceTransformer(
	br []config.NameBackReferences) (Transformer, error) {
	if br == nil {
		return nil, errors.New("backrefs not expected to be nil")
	}
	return &nameReferenceTransformer{backRefs: br}, nil
}

// Transform does the fields update according to fieldSpecs.
// The old name is in the key in the map and the new name is in the object
// associated with the key. e.g. if <k, v> is one of the key-value pair in the map,
// then the old name is k.Name and the new name is v.GetName()
func (o *nameReferenceTransformer) Transform(m resmap.ResMap) error {
	for id := range m {
		objMap := m[id].Map()
		for _, backRef := range o.backRefs {
			for _, path := range backRef.FieldSpecs {
				if !id.Gvk().IsSelected(&path.Gvk) {
					continue
				}
				err := mutateField(objMap, path.PathSlice(), path.CreateIfNotPresent,
					o.updateNameReference(backRef.Gvk, m.FilterBy(id)))
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func (o *nameReferenceTransformer) updateNameReference(
	k gvk.Gvk, m resmap.ResMap) func(in interface{}) (interface{}, error) {
	return func(in interface{}) (interface{}, error) {
		s, ok := in.(string)
		if !ok {
			return nil, fmt.Errorf("%#v is expectd to be %T", in, s)
		}

		for id, res := range m {
			if !id.Gvk().IsSelected(&k) {
				continue
			}
			if id.Name() == s {
				err := o.detectConflict(id, m, s)
				if err != nil {
					return nil, err
				}
				return res.GetName(), nil
			}
		}
		return in, nil
	}
}

func (o *nameReferenceTransformer) detectConflict(id resid.ResId, m resmap.ResMap, name string) error {
	matchedIds := m.FindByGVKN(id)
	if len(matchedIds) > 1 {
		return fmt.Errorf("detected conflicts when resolving name references %s:\n%v", name, matchedIds)
	}
	return nil
}
