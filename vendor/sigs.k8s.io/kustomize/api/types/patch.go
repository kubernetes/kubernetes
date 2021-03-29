// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// Patch represent either a Strategic Merge Patch or a JSON patch
// and its targets.
// The content of the patch can either be from a file
// or from an inline string.
type Patch struct {
	// Path is a relative file path to the patch file.
	Path string `json:"path,omitempty" yaml:"path,omitempty"`

	// Patch is the content of a patch.
	Patch string `json:"patch,omitempty" yaml:"patch,omitempty"`

	// Target points to the resources that the patch is applied to
	Target *Selector `json:"target,omitempty" yaml:"target,omitempty"`
}

// Equals return true if p equals o.
func (p *Patch) Equals(o Patch) bool {
	targetEqual := (p.Target == o.Target) ||
		(p.Target != nil && o.Target != nil && *p.Target == *o.Target)
	return p.Path == o.Path &&
		p.Patch == o.Patch &&
		targetEqual
}
