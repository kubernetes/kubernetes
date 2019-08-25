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

package plugins_test

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	. "sigs.k8s.io/kustomize/pkg/plugins"
)

// Regression coverage over compiler behavior.
func TestCompiler(t *testing.T) {
	configRoot, err := ioutil.TempDir("", "kustomize-compiler-test")
	if err != nil {
		t.Errorf("failed to make temp dir: %v", err)
	}
	srcRoot, err := DefaultSrcRoot()
	if err != nil {
		t.Error(err)
	}
	c := NewCompiler(srcRoot, configRoot)
	if configRoot != c.ObjRoot() {
		t.Errorf("unexpected objRoot %s", c.ObjRoot())
	}

	expectObj := filepath.Join(
		c.ObjRoot(),
		"someteam.example.com", "v1", "dateprefixer", "DatePrefixer.so")
	if FileExists(expectObj) {
		t.Errorf("obj file should not exist yet: %s", expectObj)
	}
	err = c.Compile("someteam.example.com", "v1", "DatePrefixer")
	if err != nil {
		t.Error(err)
	}
	if !RecentFileExists(expectObj) {
		t.Errorf("didn't find expected obj file %s", expectObj)
	}

	expectObj = filepath.Join(
		c.ObjRoot(),
		"builtin", "", "secretgenerator", "SecretGenerator.so")
	if FileExists(expectObj) {
		t.Errorf("obj file should not exist yet: %s", expectObj)
	}
	err = c.Compile("builtin", "", "SecretGenerator")
	if err != nil {
		t.Error(err)
	}
	if !RecentFileExists(expectObj) {
		t.Errorf("didn't find expected obj file %s", expectObj)
	}

	err = os.RemoveAll(c.ObjRoot())
	if err != nil {
		t.Errorf(
			"removing temp dir: %s %v", c.ObjRoot(), err)
	}
	if FileExists(expectObj) {
		t.Errorf("cleanup failed; still see: %s", expectObj)
	}
}
