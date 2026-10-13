// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// PatchArgs represent set of options on resources of a patch.
type PatchArgs struct {
	// AllowNameChange allows name changes to the resource.
	AllowNameChange bool `json:"allowNameChange,omitempty" yaml:"allowNameChange,omitempty"`

	// AllowKindChange allows kind changes to the resource.
	AllowKindChange bool `json:"allowKindChange,omitempty" yaml:"allowKindChange,omitempty"`
}
