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

package typed_test

import (
	"io/ioutil"
	"path/filepath"
	"strings"
	"testing"

	yaml "gopkg.in/yaml.v2"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

func testdata(file string) string {
	return filepath.Join("..", "internal", "testdata", file)
}

func read(file string) []byte {
	obj, err := ioutil.ReadFile(file)
	if err != nil {
		panic(err)
	}
	return obj
}

func lastPart(s string) string {
	return s[strings.LastIndex(s, ".")+1:]
}

func BenchmarkConvertUnstructured(b *testing.B) {
	tests := []struct {
		typename string
		obj      []byte
	}{
		{
			typename: "io.k8s.api.core.v1.Pod",
			obj:      read(testdata("pod.yaml")),
		},
		{
			typename: "io.k8s.api.core.v1.Node",
			obj:      read(testdata("node.yaml")),
		},
		{
			typename: "io.k8s.api.core.v1.Endpoints",
			obj:      read(testdata("endpoints.yaml")),
		},
		{
			typename: "io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1beta1.CustomResourceDefinition",
			obj:      read(testdata("prometheus-crd.yaml")),
		},
	}

	s, err := ioutil.ReadFile(testdata("k8s-schema.yaml"))
	if err != nil {
		b.Fatal(err)
	}
	parser, err := typed.NewParser(typed.YAMLObject(s))
	if err != nil {
		b.Fatal(err)
	}

	for _, test := range tests {
		pt := parser.Type(test.typename)

		obj := map[string]interface{}{}
		if err := yaml.Unmarshal(test.obj, &obj); err != nil {
			b.Fatal(err)
		}

		b.Run(lastPart(test.typename), func(b *testing.B) {
			b.Run("From", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					if _, err := pt.FromUnstructured(obj); err != nil {
						b.Fatal(err)
					}
				}
			})
			b.Run("To", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()

				// This test only exists to show that
				// this operation is now instantaneous
				// since it's not needed anymore.
			})
		})
	}
}
