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
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/json"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
)

func TestPrune(t *testing.T) {
	tests := []struct {
		name           string
		json           string
		isResourceRoot bool
		schema         *structuralschema.Structural
		expectedObject string
		expectedPruned []string
	}{
		{name: "empty", json: "null", expectedObject: "null"},
		{name: "scalar", json: "4", schema: &structuralschema.Structural{}, expectedObject: "4"},
		{name: "scalar array", json: "[1,2]", schema: &structuralschema.Structural{
			Items: &structuralschema.Structural{},
		}, expectedObject: "[1,2]"},
		{name: "object array", json: `[{"a":1},{"b":1},{"a":1,"b":2,"c":3}]`, schema: &structuralschema.Structural{
			Items: &structuralschema.Structural{
				Properties: map[string]structuralschema.Structural{
					"a": {},
					"c": {},
				},
			},
		}, expectedObject: `[{"a":1},{},{"a":1,"c":3}]`, expectedPruned: []string{"[1].b", "[2].b"}},
		{name: "object array with nil schema", json: `[{"a":1},{"b":1},{"a":1,"b":2,"c":3}]`, expectedObject: `[{},{},{}]`,
			expectedPruned: []string{"[0].a", "[1].b", "[2].a", "[2].b", "[2].c"}},
		{name: "object array object", json: `{"array":[{"a":1},{"b":1},{"a":1,"b":2,"c":3}],"unspecified":{"a":1},"specified":{"a":1,"b":2,"c":3}}`, schema: &structuralschema.Structural{
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
		}, expectedObject: `{"array":[{"a":1},{},{"a":1,"c":3}],"specified":{"a":1,"c":3}}`,
			expectedPruned: []string{"array[1].b", "array[2].b", "specified.b", "unspecified"}},
		{name: "nested x-kubernetes-preserve-unknown-fields", json: `
{
  "unspecified":"bar",
  "alpha": "abc",
  "beta": 42.0,
  "unspecifiedObject": {"unspecified": "bar"},
  "pruning": {
     "unspecified": "bar",
     "unspecifiedObject": {"unspecified": "bar"},
     "pruning": {"unspecified": "bar"},
     "apiVersion": "unknown",
     "preserving": {"unspecified": "bar"}
  },
  "preserving": {
     "unspecified": "bar",
     "unspecifiedObject": {"unspecified": "bar"},
     "pruning": {"unspecified": "bar"},
     "preserving": {"unspecified": "bar"},
     "preservingUnknownType": [{"foo":true},{"bar":true}]
  },
  "preservingAdditionalPropertiesNotInheritingXPreserveUnknownFields": {
     "foo": {
        "specified": {"unspecified":"bar"},
        "unspecified": "bar"
     }
  },
  "preservingAdditionalPropertiesKeyPruneValues": {
     "foo": {
        "specified": {"unspecified":"bar"},
        "unspecified": "bar"
     }
  }
}
`, schema: &structuralschema.Structural{
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
						"preservingUnknownType": {
							Generic:    structuralschema.Generic{Type: ""},
							Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
						},
						"pruning": {
							Generic: structuralschema.Generic{Type: "object"},
						},
					},
				},
				"preservingAdditionalPropertiesNotInheritingXPreserveUnknownFields": {
					// this x-kubernetes-preserve-unknown-fields is not inherited by the schema inside of additionalProperties
					Extensions: structuralschema.Extensions{XPreserveUnknownFields: true},
					Generic: structuralschema.Generic{
						Type: "object",
					},
					AdditionalProperties: &structuralschema.StructuralOrBool{
						Structural: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: "object"},
							Properties: map[string]structuralschema.Structural{
								"specified": {Generic: structuralschema.Generic{Type: "object"}},
							},
						},
					},
				},
				"preservingAdditionalPropertiesKeyPruneValues": {
					Generic: structuralschema.Generic{
						Type: "object",
					},
					AdditionalProperties: &structuralschema.StructuralOrBool{
						Structural: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: "object"},
							Properties: map[string]structuralschema.Structural{
								"specified": {Generic: structuralschema.Generic{Type: "object"}},
							},
						},
					},
				},
			},
		}, expectedObject: `
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
     "preserving": {"unspecified": "bar"},
     "preservingUnknownType": [{"foo":true},{"bar":true}]
  },
  "preservingAdditionalPropertiesNotInheritingXPreserveUnknownFields": {
     "foo": {
        "specified": {}
     }
  },
  "preservingAdditionalPropertiesKeyPruneValues": {
     "foo": {
        "specified": {}
     }
  }
}
`, expectedPruned: []string{"preserving.pruning.unspecified", "preservingAdditionalPropertiesKeyPruneValues.foo.specified.unspecified", "preservingAdditionalPropertiesKeyPruneValues.foo.unspecified", "preservingAdditionalPropertiesNotInheritingXPreserveUnknownFields.foo.specified.unspecified", "preservingAdditionalPropertiesNotInheritingXPreserveUnknownFields.foo.unspecified", "pruning.apiVersion", "pruning.pruning.unspecified", "pruning.unspecified", "pruning.unspecifiedObject"}},
		{name: "additionalProperties with schema", json: `{"a":1,"b":1,"c":{"a":1,"b":2,"c":{"a":1}}}`, schema: &structuralschema.Structural{
			Properties: map[string]structuralschema.Structural{
				"a": {},
				"c": {
					AdditionalProperties: &structuralschema.StructuralOrBool{
						Structural: &structuralschema.Structural{
							Generic: structuralschema.Generic{
								Type: "integer",
							},
						},
					},
				},
			},
		}, expectedObject: `{"a":1,"c":{"a":1,"b":2,"c":{}}}`,
			expectedPruned: []string{"b", "c.c.a"}},
		{name: "additionalProperties with bool", json: `{"a":1,"b":1,"c":{"a":1,"b":2,"c":{"a":1, "apiVersion": "unknown"}}}`, schema: &structuralschema.Structural{
			Properties: map[string]structuralschema.Structural{
				"a": {},
				"c": {
					AdditionalProperties: &structuralschema.StructuralOrBool{
						Bool: false,
					},
				},
			},
		}, expectedObject: `{"a":1,"c":{"a":1,"b":2,"c":{}}}`,
			expectedPruned: []string{"b", "c.c.a", "c.c.apiVersion"}},
		{name: "x-kubernetes-embedded-resource", json: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "unspecified":"bar",
  "pruned": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "unspecified": "bar"
    }
  },
  "preserving": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "unspecified": "bar"
    }
  },
  "nested": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "unspecified": "bar",
      "embedded": {
        "apiVersion": "foo/v1",
        "kind": "Foo",
        "unspecified": "bar",
        "metadata": {
          "name": "instance",
          "unspecified": "bar"
        },
        "spec": {
          "unspecified": "bar"
        }
      }
    }
  }
}
`, schema: &structuralschema.Structural{
			Generic: structuralschema.Generic{Type: "object"},
			Properties: map[string]structuralschema.Structural{
				"pruned": {
					Generic: structuralschema.Generic{Type: "object"},
					Extensions: structuralschema.Extensions{
						XEmbeddedResource: true,
					},
					Properties: map[string]structuralschema.Structural{
						"spec": {
							Generic: structuralschema.Generic{Type: "object"},
						},
					},
				},
				"preserving": {
					Generic: structuralschema.Generic{Type: "object"},
					Extensions: structuralschema.Extensions{
						XEmbeddedResource:      true,
						XPreserveUnknownFields: true,
					},
				},
				"nested": {
					Generic: structuralschema.Generic{Type: "object"},
					Extensions: structuralschema.Extensions{
						XEmbeddedResource: true,
					},
					Properties: map[string]structuralschema.Structural{
						"spec": {
							Generic: structuralschema.Generic{Type: "object"},
							Properties: map[string]structuralschema.Structural{
								"embedded": {
									Generic: structuralschema.Generic{Type: "object"},
									Extensions: structuralschema.Extensions{
										XEmbeddedResource: true,
									},
									Properties: map[string]structuralschema.Structural{
										"spec": {
											Generic: structuralschema.Generic{Type: "object"},
										},
									},
								},
							},
						},
					},
				},
			},
		}, expectedObject: `
{
  "pruned": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
    }
  },
  "preserving": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "unspecified": "bar"
    }
  },
  "nested": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "embedded": {
        "apiVersion": "foo/v1",
        "kind": "Foo",
        "metadata": {
          "name": "instance",
          "unspecified": "bar"
        },
        "spec": {
        }
      }
    }
  }
}
`, expectedPruned: []string{"nested.spec.embedded.spec.unspecified", "nested.spec.embedded.unspecified", "nested.spec.unspecified", "nested.unspecified", "pruned.spec.unspecified", "pruned.unspecified", "unspecified"}},
		{name: "x-kubernetes-embedded-resource, with root=true", json: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "namespace": "myns",
    "labels":{"foo":"bar"},
    "unspecified": "bar"
  },
  "unspecified":"bar",
  "pruned": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "namespace": "myns",
      "labels":{"foo":"bar"},
      "unspecified": "bar"
    },
    "spec": {
      "unspecified": "bar"
    }
  },
  "preserving": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "namespace": "myns",
      "labels":{"foo":"bar"},
      "unspecified": "bar"
    },
    "spec": {
      "unspecified": "bar"
    }
  },
  "nested": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "namespace": "myns",
      "labels":{"foo":"bar"},
      "unspecified": "bar"
    },
    "spec": {
      "unspecified": "bar",
      "embedded": {
        "apiVersion": "foo/v1",
        "kind": "Foo",
        "unspecified": "bar",
        "metadata": {
          "name": "instance",
          "namespace": "myns",
          "labels":{"foo":"bar"},
          "unspecified": "bar"
        },
        "spec": {
          "unspecified": "bar"
        }
      }
    }
  }
}
`, isResourceRoot: true, schema: &structuralschema.Structural{
			Generic: structuralschema.Generic{Type: "object"},
			Properties: map[string]structuralschema.Structural{
				"metadata": {
					Generic: structuralschema.Generic{Type: "object"},
				},
				"pruned": {
					Generic: structuralschema.Generic{Type: "object"},
					Extensions: structuralschema.Extensions{
						XEmbeddedResource: true,
					},
					Properties: map[string]structuralschema.Structural{
						"metadata": {
							Generic: structuralschema.Generic{Type: "object"},
						},
						"spec": {
							Generic: structuralschema.Generic{Type: "object"},
						},
					},
				},
				"preserving": {
					Generic: structuralschema.Generic{Type: "object"},
					Extensions: structuralschema.Extensions{
						XEmbeddedResource:      true,
						XPreserveUnknownFields: true,
					},
				},
				"nested": {
					Generic: structuralschema.Generic{Type: "object"},
					Extensions: structuralschema.Extensions{
						XEmbeddedResource: true,
					},
					Properties: map[string]structuralschema.Structural{
						"spec": {
							Generic: structuralschema.Generic{Type: "object"},
							Properties: map[string]structuralschema.Structural{
								"embedded": {
									Generic: structuralschema.Generic{Type: "object"},
									Extensions: structuralschema.Extensions{
										XEmbeddedResource: true,
									},
									Properties: map[string]structuralschema.Structural{
										"metadata": {
											Generic: structuralschema.Generic{Type: "object"},
										},
										"spec": {
											Generic: structuralschema.Generic{Type: "object"},
										},
									},
								},
							},
						},
					},
				},
			},
		}, expectedObject: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "namespace": "myns",
    "labels": {"foo": "bar"},
    "unspecified": "bar"
  },
  "pruned": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "metadata": {
      "name": "instance",
      "namespace": "myns",
      "labels": {"foo": "bar"},
      "unspecified": "bar"
    },
    "spec": {
    }
  },
  "preserving": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "namespace": "myns",
      "labels": {"foo": "bar"},
      "unspecified": "bar"
    },
    "spec": {
      "unspecified": "bar"
    }
  },
  "nested": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "metadata": {
      "name": "instance",
      "namespace": "myns",
      "labels": {"foo": "bar"},
      "unspecified": "bar"
    },
    "spec": {
      "embedded": {
        "apiVersion": "foo/v1",
        "kind": "Foo",
        "metadata": {
          "name": "instance",
          "namespace": "myns",
          "labels": {"foo": "bar"},
          "unspecified": "bar"
        },
        "spec": {
        }
      }
    }
  }
}
`, expectedPruned: []string{"nested.spec.embedded.spec.unspecified", "nested.spec.embedded.unspecified", "nested.spec.unspecified", "nested.unspecified", "pruned.spec.unspecified", "pruned.unspecified", "unspecified"}},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var in interface{}
			if err := json.Unmarshal([]byte(tt.json), &in); err != nil {
				t.Fatal(err)
			}

			var expectedObject interface{}
			if err := json.Unmarshal([]byte(tt.expectedObject), &expectedObject); err != nil {
				t.Fatal(err)
			}

			pruned := PruneWithOptions(in, tt.schema, tt.isResourceRoot, structuralschema.UnknownFieldPathOptions{
				TrackUnknownFieldPaths: true,
			})
			if !reflect.DeepEqual(in, expectedObject) {
				var buf bytes.Buffer
				enc := json.NewEncoder(&buf)
				enc.SetIndent("", "  ")
				err := enc.Encode(in)
				if err != nil {
					t.Fatalf("unexpected result mashalling error: %v", err)
				}
				t.Errorf("expected object: %s\ngot: %s\ndiff: %s", tt.expectedObject, buf.String(), cmp.Diff(expectedObject, in))
			}
			if !reflect.DeepEqual(pruned, tt.expectedPruned) {
				t.Errorf("expected pruned:\n\t%v\ngot:\n\t%v\n", strings.Join(tt.expectedPruned, "\n\t"), strings.Join(pruned, "\n\t"))
			}

			// now check that pruned is empty when TrackUnknownFieldPaths is false
			emptyPruned := PruneWithOptions(in, tt.schema, tt.isResourceRoot, structuralschema.UnknownFieldPathOptions{})
			if !reflect.DeepEqual(in, expectedObject) {
				var buf bytes.Buffer
				enc := json.NewEncoder(&buf)
				enc.SetIndent("", "  ")
				err := enc.Encode(in)
				if err != nil {
					t.Fatalf("unexpected result mashalling error: %v", err)
				}
				t.Errorf("expected object: %s\ngot: %s\ndiff: %s", tt.expectedObject, buf.String(), cmp.Diff(expectedObject, in))
			}
			if len(emptyPruned) > 0 {
				t.Errorf("unexpectedly returned pruned fields: %v", emptyPruned)
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
		Prune(instances[i], schema, true)
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
		//nolint:staticcheck //iccheck // SA4010 the result of append is never used, it's acceptable since in benchmark testing.
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
