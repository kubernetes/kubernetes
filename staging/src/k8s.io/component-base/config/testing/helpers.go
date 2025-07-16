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
	"os"
	"path/filepath"
	"testing"

	"github.com/google/go-cmp/cmp" //nolint:depguard
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

// TestCase defines a testcase for roundtrip and defaulting tests
type TestCase struct {
	name, in, out string
	codec         runtime.Codec
}

// RunTestsOnYAMLData decodes the yaml file from specified path, encodes the object and matches
// with expected yaml in specified path
func RunTestsOnYAMLData(t *testing.T, tests []TestCase) {
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			roundTrip(t, tc)
		})
	}
}

func decodeYAML(t *testing.T, path string, codec runtime.Codec) runtime.Object {
	content, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// decode to internal type
	object, err := runtime.Decode(codec, content)
	if err != nil {
		t.Fatal(err)
	}

	return object
}

func getCodecForGV(codecs serializer.CodecFactory, gv schema.GroupVersion) (runtime.Codec, error) {
	mediaType := runtime.ContentTypeYAML
	serializerInfo, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
	if !ok {
		return nil, fmt.Errorf("unable to locate encoder -- %q is not a supported media type", mediaType)
	}
	codec := codecs.CodecForVersions(serializerInfo.Serializer, codecs.UniversalDeserializer(), gv, nil)
	return codec, nil
}

func matchOutputFile(t *testing.T, actual []byte, expectedFilePath string) {
	expected, err := os.ReadFile(expectedFilePath)
	if err != nil && !os.IsNotExist(err) {
		t.Fatalf("couldn't read test data: %v", err)
	}

	needsUpdate := false
	const updateEnvVar = "UPDATE_COMPONENTCONFIG_FIXTURE_DATA"

	if os.IsNotExist(err) {
		needsUpdate = true
		if os.Getenv(updateEnvVar) != "true" {
			t.Error("couldn't find test data")
		}
	} else {
		if !bytes.Equal(expected, actual) {
			t.Errorf("Output does not match expected, diff (- want, + got):\n%s\n",
				cmp.Diff(string(expected), string(actual)))
			needsUpdate = true
		}
	}
	if needsUpdate {
		if os.Getenv(updateEnvVar) == "true" {
			if err := os.MkdirAll(filepath.Dir(expectedFilePath), 0755); err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(expectedFilePath, actual, 0644); err != nil {
				t.Fatal(err)
			}
			t.Error("wrote expected test data... verify, commit, and rerun tests")
		} else {
			t.Errorf("if the diff is expected because of a new type or a new field, "+
				"re-run with %s=true to update the compatibility data or generate missing files", updateEnvVar)
		}
	}
}
