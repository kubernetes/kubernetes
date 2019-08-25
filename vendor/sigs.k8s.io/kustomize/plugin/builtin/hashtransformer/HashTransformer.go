// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

//go:generate go run sigs.k8s.io/kustomize/cmd/pluginator
package main

import (
	"fmt"

	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resmap"
)

type plugin struct {
	hasher ifc.KunstructuredHasher
}

//noinspection GoUnusedGlobalVariable
var KustomizePlugin plugin

func (p *plugin) Config(
	ldr ifc.Loader, rf *resmap.Factory, config []byte) (err error) {
	p.hasher = rf.RF().Hasher()
	return nil
}

// Transform appends hash to generated resources.
func (p *plugin) Transform(m resmap.ResMap) error {
	for _, res := range m.Resources() {
		if res.NeedHashSuffix() {
			h, err := p.hasher.Hash(res)
			if err != nil {
				return err
			}
			res.SetName(fmt.Sprintf("%s-%s", res.GetName(), h))
		}
	}
	return nil
}
