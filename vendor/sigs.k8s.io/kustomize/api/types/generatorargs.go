// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// GeneratorArgs contains arguments common to ConfigMap and Secret generators.
type GeneratorArgs struct {
	// Namespace for the configmap, optional
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`

	// Name - actually the partial name - of the generated resource.
	// The full name ends up being something like
	// NamePrefix + this.Name + hash(content of generated resource).
	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	// Behavior of generated resource, must be one of:
	//   'create': create a new one
	//   'replace': replace the existing one
	//   'merge': merge with the existing one
	Behavior string `json:"behavior,omitempty" yaml:"behavior,omitempty"`

	// KvPairSources for the generator.
	KvPairSources `json:",inline,omitempty" yaml:",inline,omitempty"`
}
