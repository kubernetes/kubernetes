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
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/json"
)

func TestPrune(t *testing.T) {
	tests := []struct {
		name           string
		json           string
		isResourceRoot bool
		schema         *structuralschema.Structural
		expected       string
	}{
		{name: "empty", json: "null", expected: "null"},
		{name: "scalar", json: "4", schema: &structuralschema.Structural{}, expected: "4"},
		{name: "scalar array", json: "[1,2]", schema: &structuralschema.Structural{
			Items: &structuralschema.Structural{},
		}, expected: "[1,2]"},
		{name: "object array", json: `[{"a":1},{"b":1},{"a":1,"b":2,"c":3}]`, schema: &structuralschema.Structural{
			Items: &structuralschema.Structural{
				Properties: map[string]structuralschema.Structural{
					"a": {},
					"c": {},
				},
			},
		}, expected: `[{"a":1},{},{"a":1,"c":3}]`},
		{name: "object array with nil schema", json: `[{"a":1},{"b":1},{"a":1,"b":2,"c":3}]`, expected: `[{},{},{}]`},
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
		}, expected: `{"array":[{"a":1},{},{"a":1,"c":3}],"specified":{"a":1,"c":3}}`},
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
     "preserving": {"unspecified": "bar"}
  },
  "preserving": {
     "unspecified": "bar",
     "unspecifiedObject": {"unspecified": "bar"},
     "pruning": {"unspecified": "bar"},
     "preserving": {"unspecified": "bar"}
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
				"preservingAdditionalPropertiesKeyPruneValues": {
					Generic: structuralschema.Generic{
						Type: "object",
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
			},
		}, expected: `
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
`},
		{name: "additionalProperties with schema", json: `{"a":1,"b":1,"c":{"a":1,"b":2,"c":{"a":1}}}`, schema: &structuralschema.Structural{
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
		}, expected: `{"a":1,"c":{"a":1,"b":2,"c":{}}}`},
		{name: "additionalProperties with bool", json: `{"a":1,"b":1,"c":{"a":1,"b":2,"c":{"a":1}}}`, schema: &structuralschema.Structural{
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
		}, expected: `{"a":1,"c":{"a":1,"b":2,"c":{}}}`},
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
		}, expected: `
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
`},
		{name: "x-kubernetes-embedded-resource, with root=true", json: `
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
`, isResourceRoot: true, schema: &structuralschema.Structural{
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
		}, expected: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
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
`},
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

			Prune(in, tt.schema, tt.isResourceRoot)
			if !reflect.DeepEqual(in, expected) {
				var buf bytes.Buffer
				enc := json.NewEncoder(&buf)
				enc.SetIndent("", "  ")
				err := enc.Encode(in)
				if err != nil {
					t.Fatalf("unexpected result mashalling error: %v", err)
				}
				t.Errorf("expected: %s\ngot: %s\ndiff: %s", tt.expected, buf.String(), diff.ObjectDiff(expected, in))
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
		//lint:ignore SA4010 the result of append is never used, it's acceptable since in benchmark testing.
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
