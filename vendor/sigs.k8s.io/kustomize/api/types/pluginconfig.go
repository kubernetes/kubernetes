// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

type HelmConfig struct {
	Enabled     bool
	Command     string
	ApiVersions []string
	KubeVersion string
}

// PluginConfig holds plugin configuration.
type PluginConfig struct {
	// PluginRestrictions distinguishes plugin restrictions.
	PluginRestrictions PluginRestrictions

	// BpLoadingOptions distinguishes builtin plugin behaviors.
	BpLoadingOptions BuiltinPluginLoadingOptions

	// FnpLoadingOptions sets the way function-based plugin behaviors.
	FnpLoadingOptions FnPluginLoadingOptions

	// HelmConfig contains metadata needed for allowing and running helm.
	HelmConfig HelmConfig
}

func EnabledPluginConfig(b BuiltinPluginLoadingOptions) (pc *PluginConfig) {
	pc = MakePluginConfig(PluginRestrictionsNone, b)
	pc.FnpLoadingOptions.EnableStar = true
	pc.HelmConfig.Enabled = true
	// If this command is not on PATH, tests needing it should skip.
	pc.HelmConfig.Command = "helmV3"
	return
}

func DisabledPluginConfig() *PluginConfig {
	return MakePluginConfig(
		PluginRestrictionsBuiltinsOnly,
		BploUseStaticallyLinked)
}

func MakePluginConfig(pr PluginRestrictions,
	b BuiltinPluginLoadingOptions) *PluginConfig {
	return &PluginConfig{
		PluginRestrictions: pr,
		BpLoadingOptions:   b,
	}
}
