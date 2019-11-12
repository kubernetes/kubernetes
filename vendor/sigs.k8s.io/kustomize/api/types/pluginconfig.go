// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// PluginConfig holds plugin configuration.
type PluginConfig struct {
	// AbsPluginHome is the home of kustomize plugins.
	// Kustomize plugin configuration files are k8s-style objects
	// containing the fields 'apiVersion' and 'kind', e.g.
	//   apiVersion: apps/v1
	//   kind: Deployment
	// When kustomize reads a plugin configuration file (as as result
	// of seeing the file name in the 'generators:' or 'transformers:'
	// field in a kustomization file), it must then locate the plugin
	// code (Go plugin or exec plugin).
	// Every kustomize plugin (its code, its tests, supporting data
	// files, etc.) must be housed in its own directory at
	//   ${AbsPluginHome}/${pluginApiVersion}/LOWERCASE(${pluginKind})
	// where
	//   - ${AbsPluginHome} is an absolute path, defined below.
	//   - ${pluginApiVersion} is taken from the plugin config file.
	//   - ${pluginKind} is taken from the plugin config file.
	// The value of AbsPluginHome can be any absolute path, but might
	// default to $XDG_CONFIG_HOME/kustomize/plugin.
	AbsPluginHome string

	// PluginRestrictions defines the plugin restriction state.
	// See type for more information.
	PluginRestrictions PluginRestrictions
}
