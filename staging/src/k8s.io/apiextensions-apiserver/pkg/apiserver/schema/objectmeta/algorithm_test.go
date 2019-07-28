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

package objectmeta

import (
	"bytes"
	"reflect"
	"testing"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/json"
)

func TestCoerce(t *testing.T) {
	tests := []struct {
		name              string
		json              string
		includeRoot       bool
		dropInvalidFields bool
		schema            *structuralschema.Structural
		expected          string
		expectedError     bool
	}{
		{name: "empty", json: "null", schema: nil, expected: "null"},
		{name: "scalar", json: "4", schema: &structuralschema.Structural{}, expected: "4"},
		{name: "scalar array", json: "[1,2]", schema: &structuralschema.Structural{
			Items: &structuralschema.Structural{},
		}, expected: "[1,2]"},
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
      "name": "instance"
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
      "name": "instance"
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
      "name": "instance"
    },
    "spec": {
      "unspecified": "bar",
      "embedded": {
        "apiVersion": "foo/v1",
        "kind": "Foo",
        "unspecified": "bar",
        "metadata": {
          "name": "instance"
        },
        "spec": {
          "unspecified": "bar"
        }
      }
    }
  }
}
`},
		{name: "x-kubernetes-embedded-resource, with includeRoot=true", json: `
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
`, includeRoot: true, schema: &structuralschema.Structural{
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
    "name": "instance"
  },
  "unspecified":"bar",
  "pruned": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance"
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
      "name": "instance"
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
      "name": "instance"
    },
    "spec": {
      "unspecified": "bar",
      "embedded": {
        "apiVersion": "foo/v1",
        "kind": "Foo",
        "unspecified": "bar",
        "metadata": {
          "name": "instance"
        },
        "spec": {
          "unspecified": "bar"
        }
      }
    }
  }
}
`},
		{name: "without name", json: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance"
  },
  "pruned": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "metadata": {
      "namespace": "kube-system"
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
				},
			},
		}, expected: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance"
  },
  "pruned": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "metadata": {
      "namespace": "kube-system"
    }
  }
}
`},
		{name: "x-kubernetes-embedded-resource, with dropInvalidFields=true", json: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance"
  },
  "pruned": {
    "apiVersion": 42,
    "kind": 42,
    "metadata": {
      "name": "instance",
	  "namespace": ["abc"],
      "labels": {
        "foo": 42
      }
    }
  }
}
`, dropInvalidFields: true, schema: &structuralschema.Structural{
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
			},
		}, expected: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance"
  },
  "pruned": {
    "metadata": {
      "name": "instance"
    }
  }
}
`},
		{name: "invalid metadata type, with dropInvalidFields=true", json: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance"
  },
  "pruned": {
    "apiVersion": 42,
    "kind": 42,
    "metadata": [42]
  }
}
`, dropInvalidFields: true, schema: &structuralschema.Structural{
			Generic: structuralschema.Generic{Type: "object"},
			Properties: map[string]structuralschema.Structural{
				"pruned": {
					Generic: structuralschema.Generic{Type: "object"},
					Extensions: structuralschema.Extensions{
						XEmbeddedResource: true,
					},
				},
			},
		}, expected: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance"
  },
  "pruned": {
    "metadata": [42]
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

			err := Coerce(nil, in, tt.schema, tt.includeRoot, tt.dropInvalidFields)
			if tt.expectedError && err == nil {
				t.Error("expected error, but did not get any")
			} else if !tt.expectedError && err != nil {
				t.Errorf("expected no error, but got: %v", err)
			} else if !reflect.DeepEqual(in, expected) {
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
