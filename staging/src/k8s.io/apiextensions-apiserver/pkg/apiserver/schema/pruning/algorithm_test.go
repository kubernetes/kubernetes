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

package pruning

import (
	"bytes"
	"reflect"
	"testing"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/json"
)

func TestPrune(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		schema   *structuralschema.Structural
		expected string
	}{
		{"empty", "null", nil, "null"},
		{"scalar", "4", &structuralschema.Structural{}, "4"},
		{"scalar array", "[1,2]", &structuralschema.Structural{
			Items: &structuralschema.Structural{},
		}, "[1,2]"},
		{"object array", `[{"a":1},{"b":1},{"a":1,"b":2,"c":3}]`, &structuralschema.Structural{
			Items: &structuralschema.Structural{
				Properties: map[string]structuralschema.Structural{
					"a": {},
					"c": {},
				},
			},
		}, `[{"a":1},{},{"a":1,"c":3}]`},
		{"object array with nil schema", `[{"a":1},{"b":1},{"a":1,"b":2,"c":3}]`, nil, `[{},{},{}]`},
		{"object array object", `{"array":[{"a":1},{"b":1},{"a":1,"b":2,"c":3}],"unspecified":{"a":1},"specified":{"a":1,"b":2,"c":3}}`, &structuralschema.Structural{
			Properties: map[string]structuralschema.Structural{
				"array": {
					Items: &structuralschema.Structural{
						Properties: map[string]structuralschema.Structural{
							"a": {},
							"c": {},
						},
					},
				},
				"specified": {
					Properties: map[string]structuralschema.Structural{
						"a": {},
						"c": {},
					},
				},
			},
		}, `{"array":[{"a":1},{},{"a":1,"c":3}],"specified":{"a":1,"c":3}}`},
		{"nested x-kubernetes-preserve-unknown-fields", `
{
  "unspecified":"bar",
  "alpha": "abc",
  "beta": 42.0,
  "unspecifiedObject": {"unspecified": "bar"},
  "pruning": {
     "unspecified": "bar",
     "unspecifiedObject": {"unspecified": "bar"},
     "pruning": {"unspecified": "bar"},
     "preserving": {"unspecified": "bar"}
  },
  "preserving": {
     "unspecified": "bar",
     "unspecifiedObject": {"unspecified": "bar"},
     "pruning": {"unspecified": "bar"},
     "preserving": {"unspecified": "bar"}
  }
}
`, &structuralschema.Structural{
			Generic:    structuralschema.Generic{Type: "object"},
			Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
			Properties: map[string]structuralschema.Structural{
				"alpha": {Generic: structuralschema.Generic{Type: "string"}},
				"beta":  {Generic: structuralschema.Generic{Type: "number"}},
				"pruning": {
					Generic: structuralschema.Generic{Type: "object"},
					Properties: map[string]structuralschema.Structural{
						"preserving": {
							Generic:    structuralschema.Generic{Type: "object"},
							Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
						},
						"pruning": {
							Generic: structuralschema.Generic{Type: "object"},
						},
					},
				},
				"preserving": {
					Generic:    structuralschema.Generic{Type: "object"},
					Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
					Properties: map[string]structuralschema.Structural{
						"preserving": {
							Generic:    structuralschema.Generic{Type: "object"},
							Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
						},
						"pruning": {
							Generic: structuralschema.Generic{Type: "object"},
						},
					},
				},
			},
		}, `
{
  "unspecified":"bar",
  "alpha": "abc",
  "beta": 42.0,
  "unspecifiedObject": {"unspecified": "bar"},
  "pruning": {
     "pruning": {},
     "preserving": {"unspecified": "bar"}
  },
  "preserving": {
     "unspecified": "bar",
     "unspecifiedObject": {"unspecified": "bar"},
     "pruning": {},
     "preserving": {"unspecified": "bar"}
  }
}
`},
		{"additionalProperties with schema", `{"a":1,"b":1,"c":{"a":1,"b":2,"c":{"a":1}}}`, &structuralschema.Structural{
			Properties: map[string]structuralschema.Structural{
				"a": {},
				"c": {
					Generic: structuralschema.Generic{
						AdditionalProperties: &structuralschema.StructuralOrBool{
							Structural: &structuralschema.Structural{
								Generic: structuralschema.Generic{
									Type: "integer",
								},
							},
						},
					},
				},
			},
		}, `{"a":1,"c":{"a":1,"b":2,"c":{}}}`},
		{"additionalProperties with bool", `{"a":1,"b":1,"c":{"a":1,"b":2,"c":{"a":1}}}`, &structuralschema.Structural{
			Properties: map[string]structuralschema.Structural{
				"a": {},
				"c": {
					Generic: structuralschema.Generic{
						AdditionalProperties: &structuralschema.StructuralOrBool{
							Bool: false,
						},
					},
				},
			},
		}, `{"a":1,"c":{"a":1,"b":2,"c":{}}}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var in interface{}
			if err := json.Unmarshal([]byte(tt.json), &in); err != nil {
				t.Fatal(err)
			}

			var expected interface{}
			if err := json.Unmarshal([]byte(tt.expected), &expected); err != nil {
				t.Fatal(err)
			}

			prune(in, tt.schema)
			if !reflect.DeepEqual(in, expected) {
				var buf bytes.Buffer
				enc := json.NewEncoder(&buf)
				enc.SetIndent("", "  ")
				err := enc.Encode(in)
				if err != nil {
					t.Fatalf("unexpected result mashalling error: %v", err)
				}
				t.Errorf("expected: %s\ngot: %s", tt.expected, buf.String())
			}
		})
	}
}

const smallInstance = `
{
  "unspecified":"bar",
  "alpha": "abc",
  "beta": 42.0,
  "unspecifiedObject": {"unspecified": "bar"},
  "pruning": {
     "pruning": {},
     "preserving": {"unspecified": "bar"}
  },
  "preserving": {
     "unspecified": "bar",
     "unspecifiedObject": {"unspecified": "bar"},
     "pruning": {},
     "preserving": {"unspecified": "bar"}
  }
}
`

func BenchmarkPrune(b *testing.B) {
	b.StopTimer()
	b.ReportAllocs()

	schema := &structuralschema.Structural{
		Generic:    structuralschema.Generic{Type: "object"},
		Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
		Properties: map[string]structuralschema.Structural{
			"alpha": {Generic: structuralschema.Generic{Type: "string"}},
			"beta":  {Generic: structuralschema.Generic{Type: "number"}},
			"pruning": {
				Generic: structuralschema.Generic{Type: "object"},
				Properties: map[string]structuralschema.Structural{
					"preserving": {
						Generic:    structuralschema.Generic{Type: "object"},
						Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
					},
					"pruning": {
						Generic: structuralschema.Generic{Type: "object"},
					},
				},
			},
			"preserving": {
				Generic:    structuralschema.Generic{Type: "object"},
				Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
				Properties: map[string]structuralschema.Structural{
					"preserving": {
						Generic:    structuralschema.Generic{Type: "object"},
						Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
					},
					"pruning": {
						Generic: structuralschema.Generic{Type: "object"},
					},
				},
			},
		},
	}

	var obj map[string]interface{}
	err := json.Unmarshal([]byte(smallInstance), &obj)
	if err != nil {
		b.Fatal(err)
	}

	instances := make([]map[string]interface{}, 0, b.N)
	for i := 0; i < b.N; i++ {
		instances = append(instances, runtime.DeepCopyJSON(obj))
	}

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		Prune(instances[i], schema)
	}
}

func BenchmarkDeepCopy(b *testing.B) {
	b.StopTimer()
	b.ReportAllocs()

	var obj map[string]interface{}
	err := json.Unmarshal([]byte(smallInstance), &obj)
	if err != nil {
		b.Fatal(err)
	}

	instances := make([]map[string]interface{}, 0, b.N)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		instances = append(instances, runtime.DeepCopyJSON(obj))
	}
}

func BenchmarkUnmarshal(b *testing.B) {
	b.StopTimer()
	b.ReportAllocs()

	instances := make([]map[string]interface{}, b.N)

	b.StartTimer()
	for i := 0; i < b.N; i++ {
		err := json.Unmarshal([]byte(smallInstance), &instances[i])
		if err != nil {
			b.Fatal(err)
		}
	}
}
