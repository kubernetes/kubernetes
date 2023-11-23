/*
Copyright 2015 The Kubernetes Authors.

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

package genutils

import (
	"testing"
)

// TestValidDir tests if the given directory path is valid.
func TestValidDir(t *testing.T) {
	_, err := OutDir("./")
	if err != nil {
		t.Fatal(err)
	}
}

// TestInvalidDir tests the behavior when provided an invalid directory path.
func TestInvalidDir(t *testing.T) {
	_, err := OutDir("./nondir")
	if err == nil {
		t.Fatal("expected an error")
	}
}

// TestNotDir tests the behavior when provided a file path instead of a directory.
func TestNotDir(t *testing.T) {
	_, err := OutDir("./genutils_test.go")
	if err == nil {
		t.Fatal("expected an error")
	}
}
