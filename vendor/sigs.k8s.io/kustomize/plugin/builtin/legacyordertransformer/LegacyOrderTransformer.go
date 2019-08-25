// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

//go:generate go run sigs.k8s.io/kustomize/cmd/pluginator
package main

import (
	"github.com/pkg/errors"
	"sigs.k8s.io/kustomize/pkg/ifc"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
	"sort"
)

// Sort the resources using an ordering defined in the Gvk class.
// This puts cluster-wide basic resources with no
// dependencies (like Namespace, StorageClass, etc.)
// first, and resources with a high number of dependencies
// (like ValidatingWebhookConfiguration) last.
type plugin struct{}

//noinspection GoUnusedGlobalVariable
var KustomizePlugin plugin

// Nothing needed for configuration.
func (p *plugin) Config(
	ldr ifc.Loader, rf *resmap.Factory, c []byte) (err error) {
	return nil
}

func (p *plugin) Transform(m resmap.ResMap) (err error) {
	resources := make([]*resource.Resource, m.Size())
	ids := m.AllIds()
	sort.Sort(resmap.IdSlice(ids))
	for i, id := range ids {
		resources[i], err = m.GetByCurrentId(id)
		if err != nil {
			return errors.Wrap(err, "expected match for sorting")
		}
	}
	m.Clear()
	for _, r := range resources {
		m.Append(r)
	}
	return nil
}
