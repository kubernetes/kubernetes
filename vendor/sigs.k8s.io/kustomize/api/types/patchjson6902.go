// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// PatchJson6902 represents a json patch for an object
// with format documented https://tools.ietf.org/html/rfc6902.
type PatchJson6902 struct {
	// PatchTarget refers to a Kubernetes object that the json patch will be
	// applied to. It must refer to a Kubernetes resource under the
	// purview of this kustomization. PatchTarget should use the
	// raw name of the object (the name specified in its YAML,
	// before addition of a namePrefix and a nameSuffix).
	Target *PatchTarget `json:"target" yaml:"target"`

	// relative file path for a json patch file inside a kustomization
	Path string `json:"path,omitempty" yaml:"path,omitempty"`

	// inline patch string
	Patch string `json:"patch,omitempty" yaml:"patch,omitempty"`
}
