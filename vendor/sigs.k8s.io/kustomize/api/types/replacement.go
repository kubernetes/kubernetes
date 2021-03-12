// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// Replacement defines how to perform a substitution
// where it is from and where it is to.
type Replacement struct {
	Source *ReplSource `json:"source" yaml:"source"`
	Target *ReplTarget `json:"target" yaml:"target"`
}

// ReplSource defines where a substitution is from
// It can from two different kinds of sources
//  - from a field of one resource
//  - from a string
type ReplSource struct {
	ObjRef   *Target `json:"objref,omitempty" yaml:"objref,omitempty"`
	FieldRef string  `json:"fieldref,omitempty" yaml:"fiedldref,omitempty"`
	Value    string  `json:"value,omitempty" yaml:"value,omitempty"`
}

// ReplTarget defines where a substitution is to.
type ReplTarget struct {
	ObjRef    *Selector `json:"objref,omitempty" yaml:"objref,omitempty"`
	FieldRefs []string  `json:"fieldrefs,omitempty" yaml:"fieldrefs,omitempty"`
}
