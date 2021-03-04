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
	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type namespaceTransformer struct {
	namespace        string
	fieldSpecsToUse  []config.FieldSpec
	fieldSpecsToSkip []config.FieldSpec
}

var _ Transformer = &namespaceTransformer{}

// NewNamespaceTransformer construct a namespaceTransformer.
func NewNamespaceTransformer(ns string, cf []config.FieldSpec) Transformer {
	if len(ns) == 0 {
		return NewNoOpTransformer()
	}
	var skip []config.FieldSpec
	for _, g := range gvk.ClusterLevelGvks() {
		skip = append(skip, config.FieldSpec{Gvk: g})
	}
	return &namespaceTransformer{
		namespace:        ns,
		fieldSpecsToUse:  cf,
		fieldSpecsToSkip: skip,
	}
}

// Transform adds the namespace.
func (o *namespaceTransformer) Transform(m resmap.ResMap) error {
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

			err := mutateField(objMap, path.PathSlice(), path.CreateIfNotPresent, func(_ interface{}) (interface{}, error) {
				return o.namespace, nil
			})
			if err != nil {
				return err
			}
			newid := id.CopyWithNewNamespace(o.namespace)
			m[newid] = mf[id]
		}

	}
	o.updateClusterRoleBinding(m)
	return nil
}

func (o *namespaceTransformer) updateClusterRoleBinding(m resmap.ResMap) {
	saMap := map[string]bool{}
	for id := range m {
		if id.Gvk().Equals(gvk.Gvk{Version: "v1", Kind: "ServiceAccount"}) {
			saMap[id.Name()] = true
		}
	}

	for id := range m {
		if id.Gvk().Kind != "ClusterRoleBinding" && id.Gvk().Kind != "RoleBinding" {
			continue
		}
		objMap := m[id].Map()
		subjects := objMap["subjects"].([]interface{})
		for i := range subjects {
			subject := subjects[i].(map[string]interface{})
			kind, foundk := subject["kind"]
			name, foundn := subject["name"]
			if !foundk || !foundn || kind.(string) != "ServiceAccount" {
				continue
			}
			// a ServiceAccount named “default” exists in every active namespace
			if name.(string) == "default" || saMap[name.(string)] {
				subject := subjects[i].(map[string]interface{})
				mutateField(subject, []string{"namespace"}, true, func(_ interface{}) (interface{}, error) {
					return o.namespace, nil
				})
				subjects[i] = subject
			}
		}
		objMap["subjects"] = subjects
	}
}
