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
	"fmt"
	"log"

	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type nameReferenceTransformer struct {
	backRefs []config.NameBackReferences
}

var _ Transformer = &nameReferenceTransformer{}

// NewNameReferenceTransformer constructs a nameReferenceTransformer
// with a given slice of NameBackReferences.
func NewNameReferenceTransformer(br []config.NameBackReferences) Transformer {
	if br == nil {
		log.Fatal("backrefs not expected to be nil")
	}
	return &nameReferenceTransformer{backRefs: br}
}

// Transform updates name references in resource A that refer to resource B,
// given that B's name may have changed.
//
// For example, a HorizontalPodAutoscaler (HPA) necessarily refers to a
// Deployment (the thing that the HPA scales). The Deployment name might change
// (e.g. prefix added), and the reference in the HPA has to be fixed.
//
// In the outer loop below, we encounter an HPA.  In scanning backrefs, we
// find that HPA refers to a Deployment.  So we find all resources in the same
// namespace as the HPA (and with the same prefix and suffix), and look through
// them to find all the Deployments with a resId that has a Name matching the
// field in HPA.  For each match, we overwrite the HPA name field with the value
// found in the Deployment's name field (the name in the raw object - the
// modified name - not the unmodified name in the resId).
//
// This assumes that the name stored in a ResId (the ResMap key) isn't modified
// by name transformers.  Name transformers should only modify the name in the
// body of the resource object (the value in the ResMap).
func (o *nameReferenceTransformer) Transform(m resmap.ResMap) error {
	// TODO: Too much looping.
	// Even more hidden loops in FilterBy,
	// updateNameReference and FindByGVKN.
	for id := range m {
		for _, backRef := range o.backRefs {
			for _, fSpec := range backRef.FieldSpecs {
				if id.Gvk().IsSelected(&fSpec.Gvk) {
					err := mutateField(
						m[id].Map(), fSpec.PathSlice(),
						fSpec.CreateIfNotPresent,
						o.updateNameReference(
							backRef.Gvk, m.FilterBy(id)))
					if err != nil {
						return err
					}
				}
			}
		}
	}
	return nil
}

func (o *nameReferenceTransformer) updateNameReference(
	backRef gvk.Gvk, m resmap.ResMap) func(in interface{}) (interface{}, error) {
	return func(in interface{}) (interface{}, error) {
		switch in.(type) {
		case string:
			s, _ := in.(string)
			for id, res := range m {
				if id.Gvk().IsSelected(&backRef) && id.Name() == s {
					matchedIds := m.FindByGVKN(id)
					// If there's more than one match, there's no way
					// to know which one to pick, so emit error.
					if len(matchedIds) > 1 {
						return nil, fmt.Errorf(
							"Multiple matches for name %s:\n  %v", id, matchedIds)
					}
					// Return transformed name of the object,
					// complete with prefixes, hashes, etc.
					return res.GetName(), nil
				}
			}
			return in, nil
		case []interface{}:
			l, _ := in.([]interface{})
			var names []string
			for _, item := range l {
				name, ok := item.(string)
				if !ok {
					return nil, fmt.Errorf("%#v is expected to be %T", item, name)
				}
				names = append(names, name)
			}
			for id, res := range m {
				indexes := indexOf(id.Name(), names)
				if id.Gvk().IsSelected(&backRef) && len(indexes) > 0 {
					matchedIds := m.FindByGVKN(id)
					if len(matchedIds) > 1 {
						return nil, fmt.Errorf(
							"Multiple matches for name %s:\n %v", id, matchedIds)
					}
					for _, index := range indexes {
						l[index] = res.GetName()
					}
					return l, nil
				}
			}
			return in, nil
		default:
			return nil, fmt.Errorf("%#v is expected to be either a string or a []interface{}", in)
		}
	}
}

func indexOf(s string, slice []string) []int {
	var index []int
	for i, item := range slice {
		if item == s {
			index = append(index, i)
		}
	}
	return index
}
