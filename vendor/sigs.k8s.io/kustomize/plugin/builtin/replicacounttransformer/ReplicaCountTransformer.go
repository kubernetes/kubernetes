// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

//go:generate go run sigs.k8s.io/kustomize/cmd/pluginator
package main

import (
	"fmt"

	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/transformers"
	"sigs.k8s.io/kustomize/pkg/transformers/config"
	"sigs.k8s.io/kustomize/pkg/types"
	"sigs.k8s.io/yaml"
)

// Find matching replicas declarations and replace the count.
// Eases the kustomization configuration of replica changes.
type plugin struct {
	Replica    types.Replica      `json:"replica,omitempty" yaml:"replica,omitempty"`
	FieldSpecs []config.FieldSpec `json:"fieldSpecs,omitempty" yaml:"fieldSpecs,omitempty"`
}

//noinspection GoUnusedGlobalVariable
var KustomizePlugin plugin

func (p *plugin) Config(
	ldr ifc.Loader, rf *resmap.Factory, c []byte) (err error) {

	p.Replica = types.Replica{}
	p.FieldSpecs = nil
	return yaml.Unmarshal(c, p)
}

func (p *plugin) Transform(m resmap.ResMap) error {
	for i, replicaSpec := range p.FieldSpecs {
		for _, res := range m.GetMatchingResourcesByOriginalId(p.createMatcher(i)) {
			err := transformers.MutateField(
				res.Map(), replicaSpec.PathSlice(),
				replicaSpec.CreateIfNotPresent, p.addReplicas)
			if err != nil {
				return err
			}
		}
	}

	return nil
}

// Match Replica.Name and FieldSpec
func (p *plugin) createMatcher(i int) resmap.IdMatcher {
	return func(r resid.ResId) bool {
		return r.Name == p.Replica.Name &&
			r.Gvk.IsSelected(&p.FieldSpecs[i].Gvk)
	}
}

func (p *plugin) addReplicas(in interface{}) (interface{}, error) {
	switch m := in.(type) {
	case int64:
		// Was already in the field.
	case map[string]interface{}:
		if len(m) != 0 {
			// A map was already in the replicas field, don't want to
			// discard this data silently.
			return nil, fmt.Errorf("%#v is expected to be %T", in, m)
		}
		// Just got added, default type is map, but we can return anything.
	default:
		return nil, fmt.Errorf("%#v is expected to be %T", in, m)
	}
	return p.Replica.Count, nil
}
