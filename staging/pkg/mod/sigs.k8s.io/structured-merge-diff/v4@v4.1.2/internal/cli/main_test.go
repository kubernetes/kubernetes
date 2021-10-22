/*
Copyright 2018 The Kubernetes Authors.

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

package cli

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"path/filepath"
	"testing"
)

type testCase struct {
	options   Options
	expectErr bool

	// if present, verify that the output matches; otherwise, output is ignored.
	expectedOutputPath string
}

func testdata(file string) string {
	return filepath.Join("..", "testdata", file)
}

func TestValidate(t *testing.T) {
	cases := []testCase{{
		options: Options{
			schemaPath:   testdata("schema.yaml"),
			validatePath: testdata("schema.yaml"),
		},
	}, {
		options: Options{
			schemaPath:   testdata("schema.yaml"),
			validatePath: testdata("bad-schema.yaml"),
		},
		expectErr: true,
	}}

	for _, tt := range cases {
		tt := tt
		t.Run(tt.options.validatePath, func(t *testing.T) {
			op, err := tt.options.Resolve()
			if err != nil {
				t.Fatal(err)
			}
			var b bytes.Buffer
			err = op.Execute(&b)
			if tt.expectErr {
				if err == nil {
					t.Error("unexpected success")
				}
			} else if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestMerge(t *testing.T) {
	cases := []testCase{{
		options: Options{
			schemaPath: testdata("schema.yaml"),
			merge:      true,
			lhsPath:    testdata("struct.yaml"),
			rhsPath:    testdata("list.yaml"),
		},
	}, {
		options: Options{
			schemaPath: testdata("schema.yaml"),
			merge:      true,
			lhsPath:    testdata("bad-scalar.yaml"),
			rhsPath:    testdata("scalar.yaml"),
		},
		expectedOutputPath: testdata("scalar.yaml"),
	}, {
		options: Options{
			schemaPath: testdata("schema.yaml"),
			merge:      true,
			lhsPath:    testdata("scalar.yaml"),
			rhsPath:    testdata("bad-scalar.yaml"),
		},
		expectedOutputPath: testdata("bad-scalar.yaml"),
	}, {
		options: Options{
			schemaPath: testdata("schema.yaml"),
			merge:      true,
			lhsPath:    testdata("struct.yaml"),
			rhsPath:    testdata("bad-schema.yaml"),
		},
		expectErr: true,
	}}

	for _, tt := range cases {
		tt := tt
		t.Run(tt.options.rhsPath, func(t *testing.T) {
			op, err := tt.options.Resolve()
			if err != nil {
				t.Fatal(err)
			}
			var b bytes.Buffer
			err = op.Execute(&b)
			if tt.expectErr {
				if err == nil {
					t.Error("unexpected success")
				}
				return
			} else if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			tt.checkOutput(t, b.Bytes())
		})
	}
}

func (tt *testCase) checkOutput(t *testing.T, got []byte) {
	if tt.expectedOutputPath == "" {
		return
	}
	want, err := ioutil.ReadFile(tt.expectedOutputPath)
	if err != nil {
		t.Fatalf("couldn't read expected output %q: %v", tt.expectedOutputPath, err)
	}

	if a, e := string(got), string(want); a != e {
		t.Errorf("output didn't match expected output: got:\n%v\nwanted:\n%v\n", a, e)
	}
}

func TestCompare(t *testing.T) {
	cases := []testCase{{
		options: Options{
			schemaPath: testdata("schema.yaml"),
			compare:    true,
			lhsPath:    testdata("struct.yaml"),
			rhsPath:    testdata("list.yaml"),
		},
	}, {
		options: Options{
			schemaPath: testdata("schema.yaml"),
			compare:    true,
			lhsPath:    testdata("scalar.yaml"),
			rhsPath:    testdata("bad-scalar.yaml"),
		},
		// Yes, this is a golden data test but it's only one and it's
		// just to make sure the command output stays sane. All the
		// actual operations are unit tested.
		expectedOutputPath: testdata("scalar-compare-output.txt"),
	}, {
		options: Options{
			schemaPath: testdata("schema.yaml"),
			compare:    true,
			lhsPath:    testdata("struct.yaml"),
			rhsPath:    testdata("bad-schema.yaml"),
		},
		expectErr: true,
	}}

	for _, tt := range cases {
		tt := tt
		t.Run(tt.options.rhsPath, func(t *testing.T) {
			op, err := tt.options.Resolve()
			if err != nil {
				t.Fatal(err)
			}
			var b bytes.Buffer
			err = op.Execute(&b)
			if tt.expectErr {
				if err == nil {
					t.Error("unexpected success")
				}
			} else if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			tt.checkOutput(t, b.Bytes())
		})
	}
}

func TestFieldSet(t *testing.T) {
	cases := []testCase{{
		options: Options{
			schemaPath: testdata("k8s-schema.yaml"),
			fieldset:   testdata("pod.yaml"),
			typeName:   "io.k8s.api.core.v1.Pod",
		},
		expectedOutputPath: testdata("podset.json"),
	}, {
		options: Options{
			schemaPath: testdata("k8s-schema.yaml"),
			fieldset:   testdata("node.yaml"),
			typeName:   "io.k8s.api.core.v1.Node",
		},
		expectedOutputPath: testdata("nodeset.json"),
	}, {
		options: Options{
			schemaPath: testdata("k8s-schema.yaml"),
			fieldset:   testdata("endpoints.yaml"),
			typeName:   "io.k8s.api.core.v1.Endpoints",
		},
		expectedOutputPath: testdata("endpointsset.json"),
	}, {
		options: Options{
			schemaPath: testdata("k8s-schema.yaml"),
			fieldset:   testdata("prometheus-crd.yaml"),
			typeName:   "io.k8s.apiextensions-apiserver.pkg.apis.apiextensions.v1beta1.CustomResourceDefinition",
		},
		expectedOutputPath: testdata("prometheus-crdset.json"),
	}}

	for _, tt := range cases {
		tt := tt
		t.Run(tt.options.typeName, func(t *testing.T) {
			op, err := tt.options.Resolve()
			if err != nil {
				t.Fatal(err)
			}
			var b bytes.Buffer
			err = op.Execute(&b)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			tt.checkOutput(t, b.Bytes())

			// Test that round tripping through unstructured will preserve ordering
			u := map[string]interface{}{}
			err = json.Unmarshal(b.Bytes(), &u)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			m, err := json.Marshal(&u)
			if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			tt.checkOutput(t, m)
		})
	}
}
