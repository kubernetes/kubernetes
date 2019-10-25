// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

// Package config provides configuration methods and constants
// for general plugins.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/spf13/pflag"
	"sigs.k8s.io/kustomize/api/pgmconfig"
	"sigs.k8s.io/kustomize/api/types"
)

const (
	// Used with Go plugins.
	PluginSymbol = "KustomizePlugin"

	// Location of builtins.
	BuiltinPluginPackage = "builtin"

	// ApiVersion of builtins.
	BuiltinPluginApiVersion = BuiltinPluginPackage

	// Domain from which kustomize code is imported, for locating
	// plugin source code under $GOPATH.
	DomainName = "sigs.k8s.io"

	// Name of directory housing all plugins.
	PluginRoot = "plugin"

	flagEnablePluginsName = "enable_alpha_plugins"
	flagEnablePluginsHelp = `enable plugins, an alpha feature.
See https://github.com/kubernetes-sigs/kustomize/blob/master/docs/plugins/README.md
`
	flagErrorFmt = `
unable to load external plugin %s because plugins disabled
specify the flag
  --%s
to %s`
)

func ActivePluginConfig() *types.PluginConfig {
	pc := DefaultPluginConfig()
	pc.Enabled = true
	return pc
}

func DefaultPluginConfig() *types.PluginConfig {
	return &types.PluginConfig{
		Enabled:       false,
		DirectoryPath: filepath.Join(configRoot(), PluginRoot),
	}
}

func NotEnabledErr(name string) error {
	return fmt.Errorf(
		flagErrorFmt,
		name,
		flagEnablePluginsName,
		flagEnablePluginsHelp)
}

func AddFlagEnablePlugins(set *pflag.FlagSet, v *bool) {
	set.BoolVar(
		v, flagEnablePluginsName,
		false, flagEnablePluginsHelp)
}

// Use https://github.com/kirsle/configdir instead?
func configRoot() string {
	dir := os.Getenv(pgmconfig.XdgConfigHome)
	if len(dir) == 0 {
		dir = filepath.Join(
			HomeDir(), pgmconfig.DefaultConfigSubdir)
	}
	return filepath.Join(dir, pgmconfig.ProgramName)
}

func HomeDir() string {
	home := os.Getenv(homeEnv())
	if len(home) > 0 {
		return home
	}
	return "~"
}

func homeEnv() string {
	if runtime.GOOS == "windows" {
		return "USERPROFILE"
	}
	return "HOME"
}
