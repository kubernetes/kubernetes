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
	"io/ioutil"
	"os"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// RunTestsOnYAMLData decodes the yaml file from specified path, encodes the object and matches
// with expected yaml in specified path
func RunTestsOnYAMLData(t *testing.T, scheme *runtime.Scheme, tests []TestCase, codecs serializer.CodecFactory) {
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			obj, err := decodeTestData(rt.in, scheme, rt.inGVK, codecs)
			if err != nil {
				t.Fatal(err)
			}

			const mediaType = runtime.ContentTypeYAML
			info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
			if !ok {
				t.Errorf("unable to locate encoder -- %q is not a supported media type", mediaType)
			}

			encoder := codecs.EncoderForVersion(info.Serializer, rt.outGV)

			actual, err := runtime.Encode(encoder, obj)
			if err != nil {
				t.Fatalf("failed to encode object: %v", err)
			}

			expected, err := ioutil.ReadFile(rt.out)
			if err != nil && !os.IsNotExist(err) {
				t.Fatalf("couldn't read test data: %v", err)
			}

			needsUpdate := false
			if os.IsNotExist(err) {
				needsUpdate = true
				t.Error("couldn't find test data")
			} else {
				if !bytes.Equal(expected, actual) {
					t.Errorf("Output does not match expected, diff (- want, + got):\n\tin: %s\n\tout: %s\n\tgroupversion: %s\n\tdiff: \n%s\n",
						rt.in, rt.out, rt.outGV.String(), cmp.Diff(string(expected), string(actual)))
					needsUpdate = true
				}
			}
			if needsUpdate {
				const updateEnvVar = "UPDATE_COMPONENTCONFIG_FIXTURE_DATA"
				if os.Getenv(updateEnvVar) == "true" {
					if err := ioutil.WriteFile(rt.out, actual, 0755); err != nil {
						t.Fatal(err)
					}
					t.Logf("wrote expected test data... verify, commit, and rerun tests")
				} else {
					t.Logf("if the diff is expected because of a new type or a new field, re-run with %s=true to update the compatibility data", updateEnvVar)
				}
			}
		})
	}
}

func decodeTestData(path string, scheme *runtime.Scheme, gvk schema.GroupVersionKind, codecs serializer.CodecFactory) (runtime.Object, error) {
	content, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}

	obj, _, err := codecs.DecoderToVersion(codecs.UniversalDecoder(), gvk.GroupVersion()).Decode(content, &gvk, nil)
	if err != nil {
		return nil, err
	}
	return obj, nil
}
