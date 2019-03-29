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
	"os"
	"path/filepath"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
)

func GetDefaultingTestCases(scheme *runtime.Scheme) []TestCase {
	cases := []TestCase{}
	for gvk := range scheme.AllKnownTypes() {
		beforeDir := fmt.Sprintf("testdata/%s/before", gvk.Kind)
		afterDir := fmt.Sprintf("testdata/%s/after", gvk.Kind)
		utilruntime.Must(filepath.Walk(beforeDir, func(path string, info os.FileInfo, err error) error {
			if err != nil {
				return err
			}
			if info.IsDir() {
				if info.Name() == "before" {
					return nil
				}
				return filepath.SkipDir
			}
			if !strings.HasSuffix(info.Name(), ".yaml") {
				return nil
			}
			cases = append(cases, TestCase{
				name:  fmt.Sprintf("default_%s_%s", gvk.Kind, info.Name()),
				in:    filepath.Join(beforeDir, info.Name()),
				inGVK: gvk,
				out:   filepath.Join(afterDir, info.Name()),
				outGV: gvk.GroupVersion(),
			})
			return nil
		}))
	}
	return cases
}
