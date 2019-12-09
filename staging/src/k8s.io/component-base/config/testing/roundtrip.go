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
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
)

// RoundTripTest runs roundtrip tests for given scheme
func RoundTripTest(t *testing.T, scheme *runtime.Scheme, codecs serializer.CodecFactory) {
	tc := GetRoundtripTestCases(scheme, nil)
	RunTestsOnYAMLData(t, scheme, tc, codecs)
}

// TestCase defines a testcase for roundtrip and defaulting tests
type TestCase struct {
	name, in, out string
	inGVK         schema.GroupVersionKind
	outGV         schema.GroupVersion
}

// GetRoundtripTestCases returns the testcases for roundtrip testing for given scheme
func GetRoundtripTestCases(scheme *runtime.Scheme, disallowMarshalGroupVersions sets.String) []TestCase {
	cases := []TestCase{}
	versionsForKind := map[schema.GroupKind][]string{}
	for gvk := range scheme.AllKnownTypes() {
		versionsForKind[gvk.GroupKind()] = append(versionsForKind[gvk.GroupKind()], gvk.Version)
	}

	for gk, versions := range versionsForKind {
		for _, vin := range versions {
			if vin == runtime.APIVersionInternal {
				continue // Don't try to deserialize the internal version
			}
			for _, vout := range versions {
				inGVK := schema.GroupVersionKind{Group: gk.Group, Version: vin, Kind: gk.Kind}
				marshalGV := schema.GroupVersion{Group: gk.Group, Version: vout}
				if disallowMarshalGroupVersions.Has(marshalGV.String()) {
					continue // Don't marshal a gv that is blacklisted
				}
				testdir := filepath.Join("testdata", gk.Kind, fmt.Sprintf("%sTo%s", vin, vout))
				utilruntime.Must(filepath.Walk(testdir, func(path string, info os.FileInfo, err error) error {
					if err != nil {
						return err
					}
					if info.IsDir() {
						if info.Name() == fmt.Sprintf("%sTo%s", vin, vout) {
							return nil
						}
						return filepath.SkipDir
					}
					if filepath.Ext(info.Name()) != ".yaml" {
						return nil
					}
					cases = append(cases, TestCase{
						name:  fmt.Sprintf("%sTo%s", vin, vout),
						in:    filepath.Join(testdir, info.Name()),
						inGVK: inGVK,
						out:   filepath.Join(testdir, fmt.Sprintf("%s.after_roundtrip", info.Name())),
						outGV: marshalGV,
					})

					return nil
				}))
			}
		}
	}
	return cases
}
