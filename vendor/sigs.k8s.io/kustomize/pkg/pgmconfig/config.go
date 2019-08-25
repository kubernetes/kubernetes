/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package commands holds the CLI glue mapping textual commands/args to method calls.
package pgmconfig

import (
	"os"
	"path/filepath"
	"runtime"
)

//noinspection GoSnakeCaseUsage
const (
	XDG_CONFIG_HOME     = "XDG_CONFIG_HOME"
	defaultConfigSubdir = ".config"
	PluginRoot          = "plugin"
)

// Use https://github.com/kirsle/configdir instead?
func ConfigRoot() string {
	dir := os.Getenv(XDG_CONFIG_HOME)
	if len(dir) == 0 {
		dir = filepath.Join(
			HomeDir(), defaultConfigSubdir)
	}
	return filepath.Join(dir, ProgramName)
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
