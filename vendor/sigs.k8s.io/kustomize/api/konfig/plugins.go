// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package konfig

import (
	"os"
	"path/filepath"
	"runtime"

	"sigs.k8s.io/kustomize/api/filesys"
	"sigs.k8s.io/kustomize/api/types"
)

const (
	// Symbol that must be used inside Go plugins.
	PluginSymbol = "KustomizePlugin"

	// Name of environment variable used to set AbsPluginHome.
	// See that variable for an explanation.
	KustomizePluginHomeEnv = "KUSTOMIZE_PLUGIN_HOME"

	// Relative path below XDG_CONFIG_HOME/kustomize to find plugins.
	// e.g. AbsPluginHome = XDG_CONFIG_HOME/kustomize/plugin
	RelPluginHome = "plugin"

	// Location of builtin plugins below AbsPluginHome.
	BuiltinPluginPackage = "builtin"

	// The value of kubernetes ApiVersion to use in configuration
	// files for builtin plugins.
	// The value for non-builtins can be anything.
	BuiltinPluginApiVersion = BuiltinPluginPackage

	// Domain from which kustomize code is imported, for locating
	// plugin source code under $GOPATH when GOPATH is defined.
	DomainName = "sigs.k8s.io"

	// Injected into plugin paths when plugins are disabled.
	// Provides a clue in flows that shouldn't happen.
	NoPluginHomeSentinal = "/No/non-builtin/plugins!"
)

func EnabledPluginConfig(b types.BuiltinPluginLoadingOptions) (*types.PluginConfig, error) {
	dir, err := DefaultAbsPluginHome(filesys.MakeFsOnDisk())
	if err != nil {
		return nil, err
	}
	return MakePluginConfig(types.PluginRestrictionsNone, b, dir), nil
}

func DisabledPluginConfig() *types.PluginConfig {
	return MakePluginConfig(
		types.PluginRestrictionsBuiltinsOnly,
		types.BploUseStaticallyLinked,
		NoPluginHomeSentinal)
}

func MakePluginConfig(
	pr types.PluginRestrictions,
	b types.BuiltinPluginLoadingOptions,
	home string) *types.PluginConfig {
	return &types.PluginConfig{
		PluginRestrictions: pr,
		AbsPluginHome:      home,
		BpLoadingOptions:   b,
	}
}

type NotedFunc struct {
	Note string
	F    func() string
}

// DefaultAbsPluginHome returns the absolute path in the given file
// system to first directory that looks like a good candidate for
// the home of kustomize plugins.
func DefaultAbsPluginHome(fSys filesys.FileSystem) (string, error) {
	return FirstDirThatExistsElseError(
		"plugin home directory", fSys, []NotedFunc{
			{
				Note: "homed in $" + KustomizePluginHomeEnv,
				F: func() string {
					return os.Getenv(KustomizePluginHomeEnv)
				},
			},
			{
				Note: "homed in $" + XdgConfigHomeEnv,
				F: func() string {
					return filepath.Join(
						os.Getenv(XdgConfigHomeEnv),
						ProgramName, RelPluginHome)
				},
			},
			{
				Note: "homed in default value of $" + XdgConfigHomeEnv,
				F: func() string {
					return filepath.Join(
						HomeDir(), XdgConfigHomeEnvDefault,
						ProgramName, RelPluginHome)
				},
			},
			{
				Note: "homed in home directory",
				F: func() string {
					return filepath.Join(
						HomeDir(), ProgramName, RelPluginHome)
				},
			},
		})
}

// FirstDirThatExistsElseError tests different path functions for
// existence, returning the first that works, else error if all fail.
func FirstDirThatExistsElseError(
	what string,
	fSys filesys.FileSystem,
	pathFuncs []NotedFunc) (string, error) {
	var nope []types.Pair
	for _, dt := range pathFuncs {
		dir := dt.F()
		if fSys.Exists(dir) {
			return dir, nil
		}
		nope = append(nope, types.Pair{Key: dt.Note, Value: dir})
	}
	return "", types.NewErrUnableToFind(what, nope)
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

func CurrentWorkingDir() string {
	// Try for full path first to be explicit.
	pwd := os.Getenv(pwdEnv())
	if len(pwd) > 0 {
		return pwd
	}
	return filesys.SelfDir
}

func pwdEnv() string {
	if runtime.GOOS == "windows" {
		return "CD"
	}
	return "PWD"
}
