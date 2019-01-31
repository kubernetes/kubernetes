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

package unstructuredparsing

import (
	"bytes"
	"io"
	"io/ioutil"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/runtime/serializer/yaml"
)

// Example yaml input
var in = `

# Comments will be removed during parsing
kind: Foo
apiVersion: v1alpha1
spec:
  foo: 1
metadata:
  name: foo

---

kind: Bar
apiVersion: v1alpha1
spec:
    bar: 2

metadata:
  name: bar

`

// The expected output - note the normalization of spacing & field order
var expected = `---
apiVersion: v1alpha1
kind: Foo
metadata:
  name: foo
spec:
  foo: 1
---
apiVersion: v1alpha1
kind: Bar
metadata:
  name: bar
spec:
  bar: 2
`

// TestUnstructuedParsing demonstrates how to parse a yaml file, using the unstructured codecs.
func TestUnstructuredParsing(t *testing.T) {
	// Build a yaml decoder with the unstructured Scheme
	yamlDecoder := yaml.NewDecodingSerializer(unstructured.UnstructuredJSONScheme)

	// Parse the objects from the yaml
	var objects []runtime.Object
	reader := json.YAMLFramer.NewFrameReader(ioutil.NopCloser(bytes.NewReader([]byte(in))))
	d := streaming.NewDecoder(reader, yamlDecoder)
	for {
		obj, _, err := d.Decode(nil, nil)
		if err != nil {
			if err == io.EOF {
				break
			}
			t.Fatalf("error during parse: %v", err)
		}
		objects = append(objects, obj)
	}

	// Build a generic serializer to yaml
	yamlSerializer := json.NewYAMLSerializer(json.DefaultMetaFactory, nil, nil)

	// Serialize the objects to yaml
	var b bytes.Buffer
	writer := json.YAMLFramer.NewFrameWriter(&b)
	for _, obj := range objects {
		if err := yamlSerializer.Encode(obj, writer); err != nil {
			t.Fatalf("error during encode: %v", err)
		}
	}

	// Compare the output to our expected output
	actual := b.String()
	if actual != expected {
		t.Logf("actual: %q", actual)
		t.Logf("expect: %q", expected)

		t.Errorf("output did not match expected output")
	}
}
