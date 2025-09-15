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

package defaulting

import (
	"bytes"
	"reflect"
	"testing"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/json"
)

func TestDefault(t *testing.T) {
	tests := []struct {
		name     string
		json     string
		schema   *structuralschema.Structural
		expected string
	}{
		{"empty", "null", nil, "null"},
		{"scalar", "4", &structuralschema.Structural{
			Generic: structuralschema.Generic{
				Default: structuralschema.JSON{Object: "foo"},
			},
		}, "4"},
		{"scalar array", "[1,2]", &structuralschema.Structural{
			Items: &structuralschema.Structural{
				Generic: structuralschema.Generic{
					Default: structuralschema.JSON{Object: "foo"},
				},
			},
		}, "[1,2]"},
		{"object array", `[{"a":1},{"b":1},{"c":1}]`, &structuralschema.Structural{
			Items: &structuralschema.Structural{
				Properties: map[string]structuralschema.Structural{
					"a": {
						Generic: structuralschema.Generic{
							Default: structuralschema.JSON{Object: "A"},
						},
					},
					"b": {
						Generic: structuralschema.Generic{
							Default: structuralschema.JSON{Object: "B"},
						},
					},
					"c": {
						Generic: structuralschema.Generic{
							Default: structuralschema.JSON{Object: "C"},
						},
					},
				},
			},
		}, `[{"a":1,"b":"B","c":"C"},{"a":"A","b":1,"c":"C"},{"a":"A","b":"B","c":1}]`},
		{"object array object", `{"array":[{"a":1},{"b":2}],"object":{"a":1},"additionalProperties":{"x":{"a":1},"y":{"b":2}}}`, &structuralschema.Structural{
			Properties: map[string]structuralschema.Structural{
				"array": {
					Items: &structuralschema.Structural{
						Properties: map[string]structuralschema.Structural{
							"a": {
								Generic: structuralschema.Generic{
									Default: structuralschema.JSON{Object: "A"},
								},
							},
							"b": {
								Generic: structuralschema.Generic{
									Default: structuralschema.JSON{Object: "B"},
								},
							},
						},
					},
				},
				"object": {
					Properties: map[string]structuralschema.Structural{
						"a": {
							Generic: structuralschema.Generic{
								Default: structuralschema.JSON{Object: "N"},
							},
						},
						"b": {
							Generic: structuralschema.Generic{
								Default: structuralschema.JSON{Object: "O"},
							},
						},
					},
				},
				"additionalProperties": {
					AdditionalProperties: &structuralschema.StructuralOrBool{
						Structural: &structuralschema.Structural{
							Properties: map[string]structuralschema.Structural{
								"a": {
									Generic: structuralschema.Generic{
										Default: structuralschema.JSON{Object: "alpha"},
									},
								},
								"b": {
									Generic: structuralschema.Generic{
										Default: structuralschema.JSON{Object: "beta"},
									},
								},
							},
						},
					},
				},
				"foo": {
					Generic: structuralschema.Generic{
						Default: structuralschema.JSON{Object: "bar"},
					},
				},
			},
		}, `{"array":[{"a":1,"b":"B"},{"a":"A","b":2}],"object":{"a":1,"b":"O"},"additionalProperties":{"x":{"a":1,"b":"beta"},"y":{"a":"alpha","b":2}},"foo":"bar"}`},
		{"empty and null", `[{},{"a":1},{"a":0},{"a":0.0},{"a":""},{"a":null},{"a":[]},{"a":{}}]`, &structuralschema.Structural{
			Items: &structuralschema.Structural{
				Properties: map[string]structuralschema.Structural{
					"a": {
						Generic: structuralschema.Generic{
							Default: structuralschema.JSON{Object: "A"},
						},
					},
				},
			},
		}, `[{"a":"A"},{"a":1},{"a":0},{"a":0.0},{"a":""},{"a":"A"},{"a":[]},{"a":{}}]`},
		{"null in nullable list", `[null]`, &structuralschema.Structural{
			Generic: structuralschema.Generic{
				Nullable: true,
			},
			Items: &structuralschema.Structural{
				Properties: map[string]structuralschema.Structural{
					"a": {
						Generic: structuralschema.Generic{
							Default: structuralschema.JSON{Object: "A"},
						},
					},
				},
			},
		}, `[null]`},
		{"null in non-nullable list", `[null]`, &structuralschema.Structural{
			Generic: structuralschema.Generic{
				Nullable: false,
			},
			Items: &structuralschema.Structural{
				Generic: structuralschema.Generic{
					Default: structuralschema.JSON{Object: "A"},
				},
			},
		}, `["A"]`},
		{"null in nullable object", `{"a": null}`, &structuralschema.Structural{
			Generic: structuralschema.Generic{},
			Properties: map[string]structuralschema.Structural{
				"a": {
					Generic: structuralschema.Generic{
						Nullable: true,
						Default:  structuralschema.JSON{Object: "A"},
					},
				},
			},
		}, `{"a": null}`},
		{"null in non-nullable object", `{"a": null}`, &structuralschema.Structural{
			Properties: map[string]structuralschema.Structural{
				"a": {
					Generic: structuralschema.Generic{
						Nullable: false,
						Default:  structuralschema.JSON{Object: "A"},
					},
				},
			},
		}, `{"a": "A"}`},
		{"null in nullable object with additionalProperties", `{"a": null}`, &structuralschema.Structural{
			AdditionalProperties: &structuralschema.StructuralOrBool{
				Structural: &structuralschema.Structural{
					Generic: structuralschema.Generic{
						Nullable: true,
						Default:  structuralschema.JSON{Object: "A"},
					},
				},
			},
		}, `{"a": null}`},
		{"null in non-nullable object with additionalProperties", `{"a": null}`, &structuralschema.Structural{
			AdditionalProperties: &structuralschema.StructuralOrBool{
				Structural: &structuralschema.Structural{
					Generic: structuralschema.Generic{
						Nullable: false,
						Default:  structuralschema.JSON{Object: "A"},
					},
				},
			},
		}, `{"a": "A"}`},
		{"null unknown field", `{"a": null}`, &structuralschema.Structural{
			AdditionalProperties: &structuralschema.StructuralOrBool{
				Bool: true,
			},
		}, `{"a": null}`},
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

			Default(in, tt.schema)
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
