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
	goruntime "runtime"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// DefaultingTest run defaulting tests for given scheme
func DefaultingTest(t *testing.T, scheme *runtime.Scheme, codecs serializer.CodecFactory) {
	cases := GetDefaultingTestCases(t, scheme, codecs)
	RunTestsOnYAMLData(t, cases)
}

// GetDefaultingTestCases returns defaulting testcases for given scheme
func GetDefaultingTestCases(t *testing.T, scheme *runtime.Scheme, codecs serializer.CodecFactory) []TestCase {
	cases := []TestCase{}
	for gvk := range scheme.AllKnownTypes() {
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		beforeDir := fmt.Sprintf("testdata/%s/before", gvk.Kind)
		afterDir := fmt.Sprintf("testdata/%s/after", gvk.Kind)

		suffix := ".yaml"
		if goruntime.GOOS != "linux" {
			suffix = "_non_linux.yaml"
		}
		outFilename := gvk.Version + suffix

		codec, err := getCodecForGV(codecs, gvk.GroupVersion())
		if err != nil {
			t.Fatal(err)
		}

		cases = append(cases, TestCase{
			name:  fmt.Sprintf("%s default_%s", gvk.Kind, gvk.Version),
			in:    filepath.Join(beforeDir, fmt.Sprintf("%s.yaml", gvk.Version)),
			out:   filepath.Join(afterDir, outFilename),
			codec: codec,
		})
	}
	return cases
}
