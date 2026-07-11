// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package krusty

import (
	"sigs.k8s.io/kustomize/api/internal/plugins/builtinhelpers"
	"sigs.k8s.io/kustomize/api/types"
)

type ReorderOption string

const (
	ReorderOptionLegacy      ReorderOption = "legacy"
	ReorderOptionNone        ReorderOption = "none"
	ReorderOptionUnspecified ReorderOption = "unspecified"
)

// Options holds high-level kustomize configuration options,
// e.g. are plugins enabled, should the loader be restricted
// to the kustomization root, etc.
type Options struct {
	// When true, sort the resources before emitting them,
	// per a particular sort order.  When false, don't do the
	// sort, and instead respect the depth-first resource input
	// order as specified by the kustomization file(s).

	// Sort the resources before emitting them. Possible values:
	// - "legacy": Use a fixed order that kustomize provides for backwards
	//   compatibility.
	// - "none": Respect the depth-first resource input order as specified by the
	//   kustomization file.
	// - "unspecified": The user didn't specify any preference. Kustomize will
	//   select the appropriate default.
	Reorder ReorderOption

	// When true, a label
	//     app.kubernetes.io/managed-by: kustomize-<version>
	// is added to all the resources in the build out.
	AddManagedbyLabel bool

	// Restrictions on what can be loaded from the file system.
	// See type definition.
	LoadRestrictions types.LoadRestrictions

	// Options related to kustomize plugins.
	PluginConfig *types.PluginConfig
}

// MakeDefaultOptions returns a default instance of Options.
func MakeDefaultOptions() *Options {
	return &Options{
		Reorder:           ReorderOptionNone,
		AddManagedbyLabel: false,
		LoadRestrictions:  types.LoadRestrictionsRootOnly,
		PluginConfig:      types.DisabledPluginConfig(),
	}
}

// GetBuiltinPluginNames returns a list of builtin plugin names
func GetBuiltinPluginNames() []string {
	var ret []string
	for k := range builtinhelpers.GeneratorFactories {
		ret = append(ret, k.String())
	}
	for k := range builtinhelpers.TransformerFactories {
		ret = append(ret, k.String())
	}
	return ret
}
