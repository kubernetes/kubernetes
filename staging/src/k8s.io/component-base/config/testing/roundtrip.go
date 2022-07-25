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
	"io/ioutil"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// RoundTripTest runs roundtrip tests for given scheme
func RoundTripTest(t *testing.T, scheme *runtime.Scheme, codecs serializer.CodecFactory) {
	tc := GetRoundtripTestCases(t, scheme, codecs)
	RunTestsOnYAMLData(t, tc)
}

// GetRoundtripTestCases returns the testcases for roundtrip testing for given scheme
func GetRoundtripTestCases(t *testing.T, scheme *runtime.Scheme, codecs serializer.CodecFactory) []TestCase {
	cases := []TestCase{}
	versionsForKind := map[schema.GroupKind][]string{}
	for gvk := range scheme.AllKnownTypes() {
		if gvk.Version != runtime.APIVersionInternal {
			versionsForKind[gvk.GroupKind()] = append(versionsForKind[gvk.GroupKind()], gvk.Version)
		}
	}

	for gk, versions := range versionsForKind {
		testdir := filepath.Join("testdata", gk.Kind, "roundtrip")
		dirs, err := ioutil.ReadDir(testdir)
		if err != nil {
			t.Fatalf("failed to read testdir %s: %v", testdir, err)
		}

		for _, dir := range dirs {
			for _, vin := range versions {
				for _, vout := range versions {
					marshalGVK := gk.WithVersion(vout)
					codec, err := getCodecForGV(codecs, marshalGVK.GroupVersion())
					if err != nil {
						t.Fatalf("failed to get codec for %v: %v", marshalGVK.GroupVersion().String(), err)
					}

					testname := dir.Name()
					cases = append(cases, TestCase{
						name:  fmt.Sprintf("%s_%sTo%s_%s", gk.Kind, vin, vout, testname),
						in:    filepath.Join(testdir, testname, vin+".yaml"),
						out:   filepath.Join(testdir, testname, vout+".yaml"),
						codec: codec,
					})
				}
			}
		}
	}
	return cases
}

func roundTrip(t *testing.T, tc TestCase) {
	object := decodeYAML(t, tc.in, tc.codec)

	// original object of internal type
	original := object

	// encode (serialize) the object using the provided codec
	data, err := runtime.Encode(tc.codec, object)
	if err != nil {
		t.Fatalf("failed to encode object: %v", err)
	}

	// ensure that the encoding should not alter the object
	if !apiequality.Semantic.DeepEqual(original, object) {
		t.Fatalf("encode altered the object, diff (- want, + got): \n%v", cmp.Diff(original, object))
	}

	// decode (deserialize) the encoded data back into an object
	obj2, err := runtime.Decode(tc.codec, data)
	if err != nil {
		t.Fatalf("failed to decode: %v", err)
	}

	// ensure that the object produced from decoding the encoded data is equal
	// to the original object
	if !apiequality.Semantic.DeepEqual(original, obj2) {
		t.Fatalf("object was not the same after roundtrip, diff (- want, + got):\n%v", cmp.Diff(object, obj2))
	}

	// match with the input file, checks if they're the same after roundtrip
	matchOutputFile(t, data, tc.out)
}
