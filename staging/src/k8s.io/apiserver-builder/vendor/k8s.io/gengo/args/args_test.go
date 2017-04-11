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

package args

import (
	"testing"

	"k8s.io/gengo/types"
)

func TestInputIncludes(t *testing.T) {
	a := &GeneratorArgs{
		InputDirs: []string{"a/b/..."},
	}
	if !a.InputIncludes(&types.Package{Path: "a/b/c"}) {
		t.Errorf("Expected /... syntax to work")
	}
	if a.InputIncludes(&types.Package{Path: "a/c/b"}) {
		t.Errorf("Expected correctness")
	}
}
