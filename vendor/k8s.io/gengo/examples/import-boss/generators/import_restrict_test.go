/*
Copyright 2016 The Kubernetes Authors.

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

package generators

import (
	"testing"
)

func TestRemoveLastDir(t *testing.T) {
	table := map[string]struct{ newPath, removedDir string }{
		"a/b/c": {"a/c", "b"},
	}
	for input, expect := range table {
		gotPath, gotRemoved := removeLastDir(input)
		if e, a := expect.newPath, gotPath; e != a {
			t.Errorf("%v: wanted %v, got %v", input, e, a)
		}
		if e, a := expect.removedDir, gotRemoved; e != a {
			t.Errorf("%v: wanted %v, got %v", input, e, a)
		}
	}
}
