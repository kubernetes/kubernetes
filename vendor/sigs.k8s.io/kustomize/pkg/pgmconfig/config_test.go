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

package pgmconfig

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestConfigDirNoXdg(t *testing.T) {
	xdg, isSet := os.LookupEnv(XDG_CONFIG_HOME)
	if isSet {
		os.Unsetenv(XDG_CONFIG_HOME)
	}
	s := ConfigRoot()
	if isSet {
		os.Setenv(XDG_CONFIG_HOME, xdg)
	}
	if !strings.HasSuffix(
		s,
		rootedPath(defaultConfigSubdir, ProgramName)) {
		t.Fatalf("unexpected config dir: %s", s)
	}
}

func rootedPath(elem ...string) string {
	return string(filepath.Separator) + filepath.Join(elem...)
}

func TestConfigDirWithXdg(t *testing.T) {
	xdg, isSet := os.LookupEnv(XDG_CONFIG_HOME)
	os.Setenv(XDG_CONFIG_HOME, rootedPath("blah"))
	s := ConfigRoot()
	if isSet {
		os.Setenv(XDG_CONFIG_HOME, xdg)
	}
	if s != rootedPath("blah", ProgramName) {
		t.Fatalf("unexpected config dir: %s", s)
	}
}
