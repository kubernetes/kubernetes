// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// Restrictions on what things can be referred to
// in a kustomization file.
//
//go:generate stringer -type=LoadRestrictions
type LoadRestrictions int

const (
	LoadRestrictionsUnknown LoadRestrictions = iota

	// Files referenced by a kustomization file must be in
	// or under the directory holding the kustomization
	// file itself.
	LoadRestrictionsRootOnly

	// The kustomization file may specify absolute or
	// relative paths to patch or resources files outside
	// its own tree.
	LoadRestrictionsNone
)
