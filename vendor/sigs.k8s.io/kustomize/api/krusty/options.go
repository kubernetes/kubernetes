// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package krusty

import (
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/api/types"
)

// Options holds high-level kustomize configuration options,
// e.g. are plugins enabled, should the loader be restricted
// to the kustomization root, etc.
type Options struct {
	// When true, sort the resources before emitting them,
	// per a particular sort order.  When false, don't do the
	// sort, and instead respect the depth-first resource input
	// order as specified by the kustomization file(s).
	DoLegacyResourceSort bool

	// Restrictions on what can be loaded from the file system.
	// See type definition.
	LoadRestrictions types.LoadRestrictions

	// Create an inventory object for pruning.
	DoPrune bool

	// Options related to kustomize plugins.
	PluginConfig *types.PluginConfig
}

// MakeDefaultOptions returns a default instance of Options.
func MakeDefaultOptions() *Options {
	return &Options{
		DoLegacyResourceSort: true,
		LoadRestrictions:     types.LoadRestrictionsRootOnly,
		DoPrune:              false,
		PluginConfig:         konfig.DisabledPluginConfig(),
	}
}
