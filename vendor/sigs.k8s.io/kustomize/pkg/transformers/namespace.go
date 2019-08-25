// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package transformers

import (
	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
)

type namespaceTransformer struct {
	namespace       string
	fieldSpecsToUse []config.FieldSpec
}

var _ Transformer = &namespaceTransformer{}

// NewNamespaceTransformer construct a namespaceTransformer.
func NewNamespaceTransformer(ns string, cf []config.FieldSpec) Transformer {
	return &namespaceTransformer{
		namespace:       ns,
		fieldSpecsToUse: cf,
	}
}

const metaNamespace = "metadata/namespace"

// Transform adds the namespace.
func (o *namespaceTransformer) Transform(m resmap.ResMap) (err error) {
	if len(o.namespace) == 0 {
		return nil
	}
	for _, r := range m.Resources() {
		id := r.OrgId()
		fs, ok := o.isSelected(id)
		if !ok {
			continue
		}
		if len(r.Map()) == 0 {
			// Don't mutate empty objects?
			continue
		}
		if doIt(id, fs) {
			err = o.changeNamespace(r, fs)
			if err != nil {
				return err
			}
		}
	}
	o.updateClusterRoleBinding(m)
	return nil
}

// Special casing metadata.namespace since
// all objects have it, even "ClusterKind" objects
// that don't exist in a namespace (the Namespace
// object itself doesn't live in a namespace).
func doIt(id resid.ResId, fs *config.FieldSpec) bool {
	return fs.Path != metaNamespace ||
		(fs.Path == metaNamespace && !id.IsClusterKind())
}

func (o *namespaceTransformer) changeNamespace(
	r *resource.Resource, fs *config.FieldSpec) error {
	return MutateField(
		r.Map(), fs.PathSlice(), fs.CreateIfNotPresent,
		func(_ interface{}) (interface{}, error) {
			return o.namespace, nil
		})
}

func (o *namespaceTransformer) isSelected(
	id resid.ResId) (*config.FieldSpec, bool) {
	for _, fs := range o.fieldSpecsToUse {
		if id.IsSelected(&fs.Gvk) {
			return &fs, true
		}
	}
	return nil, false
}

func (o *namespaceTransformer) updateClusterRoleBinding(m resmap.ResMap) {
	srvAccount := gvk.Gvk{Version: "v1", Kind: "ServiceAccount"}
	saMap := map[string]bool{}
	for _, id := range m.AllIds() {
		if id.Gvk.Equals(srvAccount) {
			saMap[id.Name] = true
		}
	}

	for _, res := range m.Resources() {
		if res.OrgId().Kind != "ClusterRoleBinding" &&
			res.OrgId().Kind != "RoleBinding" {
			continue
		}
		objMap := res.Map()
		subjects, ok := objMap["subjects"].([]interface{})
		if subjects == nil || !ok {
			continue
		}
		for i := range subjects {
			subject := subjects[i].(map[string]interface{})
			kind, foundk := subject["kind"]
			name, foundn := subject["name"]
			if !foundk || !foundn || kind.(string) != srvAccount.Kind {
				continue
			}
			// a ServiceAccount named “default” exists in every active namespace
			if name.(string) == "default" || saMap[name.(string)] {
				subject := subjects[i].(map[string]interface{})
				MutateField(
					subject, []string{"namespace"},
					true, func(_ interface{}) (interface{}, error) {
						return o.namespace, nil
					})
				subjects[i] = subject
			}
		}
		objMap["subjects"] = subjects
	}
}
