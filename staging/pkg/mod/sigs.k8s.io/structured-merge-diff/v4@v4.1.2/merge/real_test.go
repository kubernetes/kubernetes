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

package merge_test

import (
	"io/ioutil"
	"path/filepath"
	"strings"
	"testing"

	. "sigs.k8s.io/structured-merge-diff/v4/internal/fixture"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
)

func testdata(file string) string {
	return filepath.Join("..", "internal", "testdata", file)
}

func read(file string) []byte {
	s, err := ioutil.ReadFile(file)
	if err != nil {
		panic(err)
	}
	return s
}

func lastPart(s string) string {
	return s[strings.LastIndex(s, ".")+1:]
}

var parser = func() Parser {
	s := read(testdata("k8s-schema.yaml"))
	parser, err := typed.NewParser(typed.YAMLObject(s))
	if err != nil {
		panic(err)
	}
	return parser
}()

func BenchmarkOperations(b *testing.B) {
	benches := []struct {
		typename string
		obj      typed.YAMLObject
	}{
		{
			typename: "io.k8s.api.core.v1.Pod",
			obj:      typed.YAMLObject(read(testdata("pod.yaml"))),
		},
		{
			typename: "io.k8s.api.core.v1.Node",
			obj:      typed.YAMLObject(read(testdata("node.yaml"))),
		},
		{
			typename: "io.k8s.api.core.v1.Endpoints",
			obj:      typed.YAMLObject(read(testdata("endpoints.yaml"))),
		},
		{
			typename: "io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1beta1.CustomResourceDefinition",
			obj:      typed.YAMLObject(read(testdata("prometheus-crd.yaml"))),
		},
	}

	for _, bench := range benches {
		b.Run(lastPart(bench.typename), func(b *testing.B) {
			tests := []struct {
				name string
				ops  []Operation
			}{
				{
					name: "Create",
					ops: []Operation{
						Update{
							Manager:    "controller",
							APIVersion: "v1",
							Object:     bench.obj,
						},
					},
				},
				{
					name: "Apply",
					ops: []Operation{
						Apply{
							Manager:    "controller",
							APIVersion: "v1",
							Object:     bench.obj,
						},
					},
				},
				{
					name: "Update",
					ops: []Operation{
						Update{
							Manager:    "controller",
							APIVersion: "v1",
							Object:     bench.obj,
						},
						Update{
							Manager:    "other-controller",
							APIVersion: "v1",
							Object:     bench.obj,
						},
					},
				},
				{
					name: "UpdateVersion",
					ops: []Operation{
						Update{
							Manager:    "controller",
							APIVersion: "v1",
							Object:     bench.obj,
						},
						Update{
							Manager:    "other-controller",
							APIVersion: "v2",
							Object:     bench.obj,
						},
					},
				},
			}
			for _, test := range tests {
				b.Run(test.name, func(b *testing.B) {
					tc := TestCase{
						Ops: test.ops,
					}
					p := SameVersionParser{T: parser.Type(bench.typename)}
					tc.PreprocessOperations(p)

					b.ReportAllocs()
					b.ResetTimer()
					for n := 0; n < b.N; n++ {
						if err := tc.Bench(p); err != nil {
							b.Fatal(err)
						}
					}
				})
			}
		})
	}
}
