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
	"bytes"
	"fmt"
	"io/ioutil"
	"testing"

	"github.com/pmezard/go-difflib/difflib"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/sets"
	configserializer "k8s.io/component-base/config/serializer"
)

type TestCase struct {
	name, in, out string
	inGVK         schema.GroupVersionKind
	outGV         schema.GroupVersion
}

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
				cases = append(cases, TestCase{
					name:  fmt.Sprintf("%sTo%s", vin, vout),
					in:    fmt.Sprintf("testdata/%s/%s.yaml", gk.Kind, vin),
					inGVK: inGVK,
					out:   fmt.Sprintf("testdata/%s/%s.yaml", gk.Kind, vout),
					outGV: marshalGV,
				})
			}
		}
	}
	return cases
}

func RunTestsOnYAMLData(t *testing.T, tests []TestCase, scheme *runtime.Scheme, codecs *serializer.CodecFactory) {
	sz := configserializer.NewStrictYAMLJSONSerializer(scheme, codecs)

	for _, rt := range tests {
		t.Run(rt.name, func(t2 *testing.T) {

			obj, err := decodeTestData(rt.in, rt.inGVK, scheme, codecs)
			if err != nil {
				t2.Fatal(err)
			}

			actual, err := sz.Encode(configserializer.ContentTypeYAML, rt.outGV, obj)
			if err != nil {
				t2.Fatal(err)
			}

			expected, err := ioutil.ReadFile(rt.out)
			if err != nil {
				t2.Fatalf("couldn't read test data")
			}

			if !bytes.Equal(expected, actual) {
				t2.Errorf("the expected and actual output differs.\n\tin: %s\n\tout: %s\n\tgroupversion: %s\n\tdiff: \n%s\n",
					rt.in, rt.out, rt.outGV.String(), DiffBytes(expected, actual))
			}
		})
	}
}

func decodeTestData(path string, gvk schema.GroupVersionKind, scheme *runtime.Scheme, codecs *serializer.CodecFactory) (runtime.Object, error) {
	sz := configserializer.NewStrictYAMLJSONSerializer(scheme, codecs)

	obj, err := scheme.New(gvk)
	if err != nil {
		return nil, err
	}

	content, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	if err := sz.DecodeInto(content, obj); err != nil {
		return nil, err
	}

	return obj, nil
}

func DiffBytes(expected, actual []byte) string {
	var diffBytes bytes.Buffer
	difflib.WriteUnifiedDiff(&diffBytes, difflib.UnifiedDiff{
		A:        difflib.SplitLines(string(expected)),
		B:        difflib.SplitLines(string(actual)),
		FromFile: "expected",
		ToFile:   "actual",
		Context:  3,
	})
	return diffBytes.String()
}

/*
TODO: Add a scheme tester that enforces a couple of things...
*/
