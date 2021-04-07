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
	// kustomize reads plugin configuration data from a file path
	// specified in the 'generators:' or 'transformers:' field of a
	// kustomization file.  kustomize must then use this data to both
	// locate the plugin and configure it.
	// Every kustomize plugin (its code, its tests, its supporting data
	// files, etc.) must be housed in its own directory at
	//   ${AbsPluginHome}/${pluginApiVersion}/LOWERCASE(${pluginKind})
	// where
	//   - ${AbsPluginHome} is an absolute path, defined below.
	//   - ${pluginApiVersion} is taken from the plugin config file.
	//   - ${pluginKind} is taken from the plugin config file.
	// The value of AbsPluginHome can be any absolute path.
	AbsPluginHome string

	// PluginRestrictions distinguishes plugin restrictions.
	PluginRestrictions PluginRestrictions

	// BpLoadingOptions distinguishes builtin plugin behaviors.
	BpLoadingOptions BuiltinPluginLoadingOptions

	// FnpLoadingOptions sets the way function-based plugin behaviors.
	FnpLoadingOptions FnPluginLoadingOptions
}
