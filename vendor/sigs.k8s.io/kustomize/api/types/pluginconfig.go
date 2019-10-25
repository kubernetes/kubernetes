// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// PluginConfig holds plugin configuration.
type PluginConfig struct {
	// DirectoryPath is an absolute path to a
	// directory containing kustomize plugins.
	// This directory may contain subdirectories
	// further categorizing plugins.
	DirectoryPath string

	// Enabled is true if plugins are enabled.
	Enabled bool
}
