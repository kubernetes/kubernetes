// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// SortOptions defines the order that kustomize outputs resources.
type SortOptions struct {
	// Order selects the ordering strategy.
	Order SortOrder `json:"order,omitempty" yaml:"order,omitempty"`
	// LegacySortOptions tweaks the sorting for the "legacy" sort ordering
	// strategy.
	LegacySortOptions *LegacySortOptions `json:"legacySortOptions,omitempty" yaml:"legacySortOptions,omitempty"`
}

// SortOrder defines different ordering strategies.
type SortOrder string

const LegacySortOrder SortOrder = "legacy"
const FIFOSortOrder SortOrder = "fifo"

// LegacySortOptions define various options for tweaking the "legacy" ordering
// strategy.
type LegacySortOptions struct {
	// OrderFirst selects the resource kinds to order first.
	OrderFirst []string `json:"orderFirst" yaml:"orderFirst"`
	// OrderLast selects the resource kinds to order last.
	OrderLast []string `json:"orderLast" yaml:"orderLast"`
}
