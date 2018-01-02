/*
Copyright 2017 The Kubernetes Authors.

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

package parser

import (
	"testing"
)

func TestImportBuildPackage(t *testing.T) {
	b := New()
	if _, err := b.importBuildPackage("fake/dep"); err != nil {
		t.Fatal(err)
	}
	if _, ok := b.buildPackages["fake/dep"]; !ok {
		t.Errorf("missing expected, but got %v", b.buildPackages)
	}

	if len(b.buildPackages) > 1 {
		// this would happen if the canonicalization failed to normalize the path
		// you'd get a k8s.io/gengo/vendor/fake/dep key too
		t.Errorf("missing one, but got %v", b.buildPackages)
	}
}

func TestCanonicalizeImportPath(t *testing.T) {
	tcs := []struct {
		name   string
		input  string
		output string
	}{
		{
			name:   "passthrough",
			input:  "github.com/foo/bar",
			output: "github.com/foo/bar",
		},
		{
			name:   "simple",
			input:  "github.com/foo/vendor/k8s.io/kubernetes/pkg/api",
			output: "k8s.io/kubernetes/pkg/api",
		},
		{
			name:   "deeper",
			input:  "github.com/foo/bar/vendor/k8s.io/kubernetes/pkg/api",
			output: "k8s.io/kubernetes/pkg/api",
		},
	}

	for _, tc := range tcs {
		actual := canonicalizeImportPath(tc.input)
		if string(actual) != tc.output {
			t.Errorf("%v: expected %q got %q", tc.name, tc.output, actual)
		}
	}
}
