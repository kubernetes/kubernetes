/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package maketmpdir

import (
	"io/ioutil"
	"os"
	"testing"
)

// makeTmpDir creates a temporary directory based on the prefix given. It returns the temporary
// directory path along with a deferable function to remove the directory. If an error occurs during
// creation t.Fatal is called and execution is halted.
func MakeTmpDir(t *testing.T, prefix string) (string, func()) {
	tmpDir, err := ioutil.TempDir(os.TempDir(), prefix)
	if err != nil {
		t.Fatal("can't make a temp dir: %v", err)
	}
	return tmpDir, func() { defer os.RemoveAll(tmpDir) }
}
