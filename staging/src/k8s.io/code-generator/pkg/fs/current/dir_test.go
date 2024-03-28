/*
Copyright 2023 The Kubernetes Authors.

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

package current_test

import (
	"k8s.io/code-generator/pkg/fs/current"
	"os"
	"strings"
	"testing"
)

func TestDir(t *testing.T) {
	t.Parallel()
	dir, err := current.Dir()
	if err != nil {
		t.Errorf("no error expected, got %v", err)
	}
	want := "pkg/fs/current"
	if !strings.Contains(dir, want) {
		t.Errorf("Wanted %#v in %#v, but not found", want, dir)
	}
	st, ferr := os.Stat(dir)
	if ferr != nil {
		t.Errorf("no error expected, got %v", ferr)
	}
	if !st.IsDir() {
		t.Errorf("Wanted %#v to be a directory", dir)
	}
}
