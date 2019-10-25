// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// GeneratorOptions modify behavior of all ConfigMap and Secret generators.
type GeneratorOptions struct {
	// Labels to add to all generated resources.
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`

	// Annotations to add to all generated resources.
	Annotations map[string]string `json:"annotations,omitempty" yaml:"annotations,omitempty"`

	// DisableNameSuffixHash if true disables the default behavior of adding a
	// suffix to the names of generated resources that is a hash of the
	// resource contents.
	DisableNameSuffixHash bool `json:"disableNameSuffixHash,omitempty" yaml:"disableNameSuffixHash,omitempty"`
}
