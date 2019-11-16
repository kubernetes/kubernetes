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

package testing

import (
	"fmt"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// DefaultingTest run defaulting tests for given scheme
func DefaultingTest(t *testing.T, scheme *runtime.Scheme, codecs serializer.CodecFactory) {
	tc := GetDefaultingTestCases(scheme)
	RunTestsOnYAMLData(t, scheme, tc, codecs)
}

// GetDefaultingTestCases returns defaulting testcases for given scheme
func GetDefaultingTestCases(scheme *runtime.Scheme) []TestCase {
	cases := []TestCase{}
	for gvk := range scheme.AllKnownTypes() {
		beforeDir := fmt.Sprintf("testdata/%s/before", gvk.Kind)
		afterDir := fmt.Sprintf("testdata/%s/after", gvk.Kind)
		filename := fmt.Sprintf("%s.yaml", gvk.Version)

		cases = append(cases, TestCase{
			name:  fmt.Sprintf("default_%s", gvk.Version),
			in:    filepath.Join(beforeDir, filename),
			inGVK: gvk,
			out:   filepath.Join(afterDir, filename),
			outGV: gvk.GroupVersion(),
		})
	}
	return cases
}
