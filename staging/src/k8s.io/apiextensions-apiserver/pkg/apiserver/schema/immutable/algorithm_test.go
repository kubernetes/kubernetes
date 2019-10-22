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

package immutable

import (
	"testing"

	structuralschema "k8s.io/apiextensions-apiserver/pkg/apiserver/schema"
	"k8s.io/apimachinery/pkg/util/json"
)

func TestImmutable(t *testing.T) {
	tests := []struct {
		name          string
		jsonOld       string
		jsonNew       string
		schema        *structuralschema.Structural
		expectedError bool
	}{
		{
			name: "non mutated",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "immutable": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "mutated": "false"
    }
  }
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "immutable": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "mutated": "false"
    }
  }
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: "object"},
				Properties: map[string]structuralschema.Structural{
					"immutable": {
						Generic: structuralschema.Generic{Type: "object"},
						Extensions: structuralschema.Extensions{
							XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
						},
						Properties: map[string]structuralschema.Structural{
							"spec": {
								Generic: structuralschema.Generic{Type: "object"},
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "mutated",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "immutable": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "mutated": "false"
    }
  }
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "immutable": {
    "apiVersion": "foo/v1",
    "kind": "Foo",
    "unspecified": "bar",
    "metadata": {
      "name": "instance",
      "unspecified": "bar"
    },
    "spec": {
      "mutated": "true"
    }
  }
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: "object"},
				Properties: map[string]structuralschema.Structural{
					"immutable": {
						Generic: structuralschema.Generic{Type: "object"},
						Extensions: structuralschema.Extensions{
							XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
						},
						Properties: map[string]structuralschema.Structural{
							"spec": {
								Generic: structuralschema.Generic{Type: "object"},
							},
						},
					},
				},
			},
			expectedError: true,
		},
		{
			name: "isMutableArrayImmutableItems (different items)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "foo",
	"bar"
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "mutated",
	"bar"
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isMutableArrayImmutableItems": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType: strPtr(structuralschema.XListTypeAtomic),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Extensions: structuralschema.Extensions{
								XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							},
						},
					},
				},
			},
			expectedError: true,
		},
		{
			name: "isMutableArrayImmutableItems (different order)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "foo",
	"bar"
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "bar",
	"foo"
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isMutableArrayImmutableItems": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType: strPtr(structuralschema.XListTypeAtomic),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Extensions: structuralschema.Extensions{
								XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							},
						},
					},
				},
			},
			expectedError: true,
		},
		{
			name: "isMutableArrayImmutableItems (append)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "foo",
	"bar"
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "foo",
	"bar",
    "1",
    "2"
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isMutableArrayImmutableItems": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType: strPtr(structuralschema.XListTypeAtomic),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Extensions: structuralschema.Extensions{
								XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isMutableArrayImmutableItems (remove)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "foo",
	"bar",
    "1",
    "2"
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "foo",
	"bar"
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isMutableArrayImmutableItems": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType: strPtr(structuralschema.XListTypeAtomic),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Extensions: structuralschema.Extensions{
								XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isMutableArrayImmutableItems (equal)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "foo",
	"bar"
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayImmutableItems": [
    "foo",
	"bar"
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isMutableArrayImmutableItems": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType: strPtr(structuralschema.XListTypeAtomic),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Extensions: structuralschema.Extensions{
								XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isMutableArrayMapImmutableItems (equal)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayMapImmutableItems": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayMapImmutableItems": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isMutableArrayMapImmutableItems": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType: strPtr(structuralschema.XListTypeMap),
							XListMapKeys: []string{
								"foo",
								"bar",
							},
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Extensions: structuralschema.Extensions{
								XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isMutableArrayMapImmutableItems (different value for same key)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayMapImmutableItems": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableArrayMapImmutableItems": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "mutated"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isMutableArrayMapImmutableItems": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType: strPtr(structuralschema.XListTypeMap),
							XListMapKeys: []string{
								"foo",
								"bar",
							},
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Extensions: structuralschema.Extensions{
								XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: true,
		},
		// TODO: isMutableArrayMapImmutableItems (append, remove)
		{
			name: "isMutableMapImmutableValues (equal)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableMapImmutableValues": {
    "foo": "a1",
    "bar": "a2"
  }
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableMapImmutableValues": {
    "foo": "a1",
    "bar": "a2"
  }
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Extensions: structuralschema.Extensions{
					XMapType: strPtr(structuralschema.XMapTypeAtomic),
				},
				Properties: map[string]structuralschema.Structural{
					"isMutableMapImmutableValues": {
						Generic: structuralschema.Generic{
							Type: structuralschema.GenericTypeObject,
							AdditionalProperties: &structuralschema.StructuralOrBool{
								Structural: &structuralschema.Structural{
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
									Extensions: structuralschema.Extensions{
										XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
									},
								},
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isMutableMapImmutableValues (different value for same key)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableMapImmutableValues": {
    "foo": "a1",
    "bar": "a2"
  }
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isMutableMapImmutableValues": {
    "foo": "a1",
    "bar": "mutated"
  }
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Extensions: structuralschema.Extensions{
					XMapType: strPtr(structuralschema.XMapTypeAtomic),
				},
				Properties: map[string]structuralschema.Structural{
					"isMutableMapImmutableValues": {
						Generic: structuralschema.Generic{
							Type: structuralschema.GenericTypeObject,
							AdditionalProperties: &structuralschema.StructuralOrBool{
								Structural: &structuralschema.Structural{
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
									Extensions: structuralschema.Extensions{
										XImmutability: strPtr(structuralschema.XImmutabilityImmutable),
									},
								},
							},
						},
					},
				},
			},
			expectedError: true,
		},
		// TODO: isMutableMapImmutableValues (append, remove)
		{
			name: "isImmutableArrayKeysMutableValues (immutable: equal)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isImmutableArrayKeysMutableValues": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType:        strPtr(structuralschema.XListTypeAtomic),
							XKeyImmutability: strPtr(structuralschema.XImmutabilityImmutable),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isImmutableArrayKeysMutableValues (immutable: different length)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    },
	{
      "foo": "c1",
      "bar": "c2",
      "value": "c3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isImmutableArrayKeysMutableValues": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType:        strPtr(structuralschema.XListTypeAtomic),
							XKeyImmutability: strPtr(structuralschema.XImmutabilityImmutable),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: true,
		},
		{
			name: "isImmutableArrayKeysMutableValues (addOnly: positive)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    },
	{
      "foo": "c1",
      "bar": "c2",
      "value": "c3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isImmutableArrayKeysMutableValues": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType:        strPtr(structuralschema.XListTypeAtomic),
							XKeyImmutability: strPtr(structuralschema.XImmutabilityAddOnly),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isImmutableArrayKeysMutableValues (addOnly: negative)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isImmutableArrayKeysMutableValues": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType:        strPtr(structuralschema.XListTypeAtomic),
							XKeyImmutability: strPtr(structuralschema.XImmutabilityAddOnly),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: true,
		},
		{
			name: "isImmutableArrayKeysMutableValues (removeOnly: positive)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isImmutableArrayKeysMutableValues": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType:        strPtr(structuralschema.XListTypeAtomic),
							XKeyImmutability: strPtr(structuralschema.XImmutabilityRemoveOnly),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isImmutableArrayKeysMutableValues (removeOnly: negative)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayKeysMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    },
	{
      "foo": "c1",
      "bar": "c2",
      "value": "c3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isImmutableArrayKeysMutableValues": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType:        strPtr(structuralschema.XListTypeAtomic),
							XKeyImmutability: strPtr(structuralschema.XImmutabilityRemoveOnly),
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: true,
		},
		{
			name: "isImmutableArrayMapMutableValues (equal)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayMapMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayMapMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isImmutableArrayMapMutableValues": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType:        strPtr(structuralschema.XListTypeMap),
							XKeyImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							XListMapKeys: []string{
								"foo",
								"bar",
							},
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: false,
		},
		{
			name: "isImmutableArrayMapMutableValues (different value for same key)",
			jsonOld: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayMapMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "a3"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			jsonNew: `
{
  "apiVersion": "foo/v1",
  "kind": "Foo",
  "metadata": {
    "name": "instance",
    "unspecified": "bar"
  },
  "isImmutableArrayMapMutableValues": [
    {
      "foo": "a1",
      "bar": "a2",
      "value": "mutated"
    },
	{
      "foo": "b1",
      "bar": "b2",
      "value": "b3"
    }
  ]
}
`,
			schema: &structuralschema.Structural{
				Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
				Properties: map[string]structuralschema.Structural{
					"isImmutableArrayMapMutableValues": {
						Generic: structuralschema.Generic{Type: structuralschema.GenericTypeArray},
						Extensions: structuralschema.Extensions{
							XListType:        strPtr(structuralschema.XListTypeMap),
							XKeyImmutability: strPtr(structuralschema.XImmutabilityImmutable),
							XListMapKeys: []string{
								"foo",
								"bar",
							},
						},
						Items: &structuralschema.Structural{
							Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
							Properties: map[string]structuralschema.Structural{
								"foo": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"bar": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
								"value": {
									Generic: structuralschema.Generic{Type: structuralschema.GenericTypeObject},
								},
							},
						},
					},
				},
			},
			expectedError: false,
		},
		// TODO: isImmutableArrayMapMutableValues (append, remove)
		// TODO: IsImmutableArraySetMutableValues
		// TODO: isImmutableMapKeysMutableValues
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var old interface{}
			var new interface{}
			if err := json.Unmarshal([]byte(tt.jsonOld), &old); err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal([]byte(tt.jsonNew), &new); err != nil {
				t.Fatal(err)
			}

			if err := Immutable(old, new, tt.schema, nil); (err != nil) != tt.expectedError {
				t.Errorf("%v does not match error expectation: %v. Got: %v", tt.name, tt.expectedError, err)
			}
		})
	}
}

func strPtr(s string) *string { return &s }

func TestIsMutableArrayImmutableItems(t *testing.T) {
	tests := []struct {
		name     string
		old      []interface{}
		new      []interface{}
		expected bool
	}{
		{
			name: "equal",
			old: []interface{}{
				"foo",
				"bar",
			},
			new: []interface{}{
				"foo",
				"bar",
			},
			expected: true,
		},
		{
			name: "different order",
			old: []interface{}{
				"bar",
				"foo",
			},
			new: []interface{}{
				"foo",
				"bar",
			},
			expected: false,
		},
		{
			name: "different items",
			old: []interface{}{
				"bar",
				"foo",
			},
			new: []interface{}{
				"mutated",
				"bar",
			},
			expected: false,
		},
		{
			name: "append",
			old: []interface{}{
				"bar",
				"foo",
			},
			new: []interface{}{
				"bar",
				"foo",
				"a",
				"b",
			},
			expected: true,
		},
		{
			name: "remove",
			old: []interface{}{
				"bar",
				"foo",
			},
			new: []interface{}{
				"bar",
			},
			expected: true,
		},
		{
			name: "new removed",
			old: []interface{}{
				"bar",
				"foo",
			},
			new:      nil,
			expected: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isListAtomicImmutableItems(tc.old, tc.new); got != tc.expected {
				t.Errorf("Expected %v, got: %v", tc.expected, got)
			}
		})
	}
}

func TestIsListMapImmutableItems(t *testing.T) {
	tests := []struct {
		name     string
		keys     []string
		old      []interface{}
		new      []interface{}
		expected bool
	}{
		{
			name: "equal",
			keys: []string{"foo", "bar"},
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			expected: true,
		},
		{
			name: "different value for same key",
			keys: []string{"foo", "bar"},
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "mutated",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			expected: false,
		},
		{
			name: "add key-value",
			keys: []string{"foo", "bar"},
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
				map[string]interface{}{
					"foo":   "c1",
					"bar":   "c2",
					"value": "c3",
				},
			},
			expected: true,
		},
		{
			name: "remove key-value",
			keys: []string{"foo", "bar"},
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
			},
			expected: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got, _ := isListMapImmutableItems(tc.keys, tc.old, tc.new); got != tc.expected {
				t.Errorf("Expected %v, got: %v", tc.expected, got)
			}
		})
	}
}

func TestIsMapImmutableValues(t *testing.T) {
	tests := []struct {
		name     string
		old      map[string]interface{}
		new      map[string]interface{}
		expected bool
	}{
		{
			name: "equal",
			old: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			new: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			expected: true,
		},
		{
			name: "different value for same key",
			old: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			new: map[string]interface{}{
				"foo": "v1",
				"bar": "mutated",
			},
			expected: false,
		},
		{
			name: "add key-value",
			old: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			new: map[string]interface{}{
				"foo":   "v1",
				"bar":   "v2",
				"other": "v3",
			},
			expected: true,
		},
		{
			name: "remove key-value",
			old: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			new: map[string]interface{}{
				"foo": "v1",
			},
			expected: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isMapImmutableValues(tc.old, tc.new); got != tc.expected {
				t.Errorf("Expected %v, got: %v", tc.expected, got)
			}
		})
	}
}

func TestIsListMapImmutableKeys(t *testing.T) {
	tests := []struct {
		name               string
		keys               []string
		old                []interface{}
		new                []interface{}
		expectedImmutable  bool
		expectedAddOnly    bool
		expectedRemoveOnly bool
	}{
		{
			name: "equal",
			keys: []string{"foo", "bar"},
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			expectedImmutable:  true,
			expectedAddOnly:    true,
			expectedRemoveOnly: true,
		},
		{
			name: "different value for same key",
			keys: []string{"foo", "bar"},
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "mutated",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			expectedImmutable:  true,
			expectedAddOnly:    true,
			expectedRemoveOnly: true,
		},
		{
			name: "add key-value",
			keys: []string{"foo", "bar"},
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
				map[string]interface{}{
					"foo":   "c1",
					"bar":   "c2",
					"value": "c3",
				},
			},
			expectedImmutable:  false,
			expectedAddOnly:    true,
			expectedRemoveOnly: false,
		},
		{
			name: "remove key-value",
			keys: []string{"foo", "bar"},
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
			},
			expectedImmutable:  false,
			expectedAddOnly:    false,
			expectedRemoveOnly: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got, _ := isListMapImmutableKeys(structuralschema.XImmutabilityImmutable, tc.keys, tc.old, tc.new); got != tc.expectedImmutable {
				t.Errorf("Expected %v, got: %v", tc.expectedImmutable, got)
			}
			if got, _ := isListMapImmutableKeys(structuralschema.XImmutabilityAddOnly, tc.keys, tc.old, tc.new); got != tc.expectedAddOnly {
				t.Errorf("Expected %v, got: %v", tc.expectedAddOnly, got)
			}
			if got, _ := isListMapImmutableKeys(structuralschema.XImmutabilityRemoveOnly, tc.keys, tc.old, tc.new); got != tc.expectedRemoveOnly {
				t.Errorf("Expected %v, got: %v", tc.expectedRemoveOnly, got)
			}
		})
	}
}

func TestIsListSetImmutableKeys(t *testing.T) {
	tests := []struct {
		name               string
		old                []interface{}
		new                []interface{}
		expectedImmutable  bool
		expectedAddOnly    bool
		expectedRemoveOnly bool
	}{
		{
			name: "equal",
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			expectedImmutable:  true,
			expectedAddOnly:    true,
			expectedRemoveOnly: true,
		},
		{
			name: "different value for same key",
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "mutated",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			expectedImmutable:  false,
			expectedAddOnly:    false,
			expectedRemoveOnly: false,
		},
		{
			name: "add key-value",
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
				map[string]interface{}{
					"foo":   "c1",
					"bar":   "c2",
					"value": "c3",
				},
			},
			expectedImmutable:  false,
			expectedAddOnly:    true,
			expectedRemoveOnly: false,
		},
		{
			name: "remove key-value",
			old: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
				map[string]interface{}{
					"foo":   "b1",
					"bar":   "b2",
					"value": "b3",
				},
			},
			new: []interface{}{
				map[string]interface{}{
					"foo":   "a1",
					"bar":   "a2",
					"value": "a3",
				},
			},
			expectedImmutable:  false,
			expectedAddOnly:    false,
			expectedRemoveOnly: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got, _ := isListSetImmutableKeys(structuralschema.XImmutabilityImmutable, tc.old, tc.new); got != tc.expectedImmutable {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityImmutable, tc.expectedImmutable, got)
			}
			if got, _ := isListSetImmutableKeys(structuralschema.XImmutabilityAddOnly, tc.old, tc.new); got != tc.expectedAddOnly {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityAddOnly, tc.expectedAddOnly, got)
			}
			if got, _ := isListSetImmutableKeys(structuralschema.XImmutabilityRemoveOnly, tc.old, tc.new); got != tc.expectedRemoveOnly {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityRemoveOnly, tc.expectedRemoveOnly, got)
			}
		})
	}
}

func TestIsMapImmutableKeys(t *testing.T) {
	tests := []struct {
		name               string
		old                map[string]interface{}
		new                map[string]interface{}
		expectedImmutable  bool
		expectedAddOnly    bool
		expectedRemoveOnly bool
	}{
		{
			name: "equal",
			old: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			new: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			expectedImmutable:  true,
			expectedAddOnly:    true,
			expectedRemoveOnly: true,
		},
		{
			name: "different value for same key",
			old: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			new: map[string]interface{}{
				"foo": "v1",
				"bar": "mutated",
			},
			expectedImmutable:  true,
			expectedAddOnly:    true,
			expectedRemoveOnly: true,
		},
		{
			name: "add key-value",
			old: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			new: map[string]interface{}{
				"foo":   "v1",
				"bar":   "v2",
				"other": "v3",
			},
			expectedImmutable:  false,
			expectedAddOnly:    true,
			expectedRemoveOnly: false,
		},
		{
			name: "remove key-value",
			old: map[string]interface{}{
				"foo": "v1",
				"bar": "v2",
			},
			new: map[string]interface{}{
				"foo": "v1",
			},
			expectedImmutable:  false,
			expectedAddOnly:    false,
			expectedRemoveOnly: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isMapImmutableKeys(structuralschema.XImmutabilityImmutable, tc.old, tc.new); got != tc.expectedImmutable {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityImmutable, tc.expectedImmutable, got)
			}
			if got := isMapImmutableKeys(structuralschema.XImmutabilityAddOnly, tc.old, tc.new); got != tc.expectedAddOnly {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityAddOnly, tc.expectedAddOnly, got)
			}
			if got := isMapImmutableKeys(structuralschema.XImmutabilityRemoveOnly, tc.old, tc.new); got != tc.expectedRemoveOnly {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityRemoveOnly, tc.expectedRemoveOnly, got)
			}
		})
	}
}

func TestIsImmutable(t *testing.T) {
	key := "foo"
	tests := []struct {
		name               string
		old                map[string]interface{}
		new                map[string]interface{}
		expectedImmutable  bool
		expectedAddOnly    bool
		expectedRemoveOnly bool
	}{
		{
			name: "equal",
			old: map[string]interface{}{
				key: map[string]interface{}{
					"foo": "v1",
					"bar": "v2",
				},
			},
			new: map[string]interface{}{
				key: map[string]interface{}{
					"foo": "v1",
					"bar": "v2",
				},
			},
			expectedImmutable:  true,
			expectedAddOnly:    true,
			expectedRemoveOnly: true,
		},
		{
			name: "mutated",
			old: map[string]interface{}{
				key: map[string]interface{}{
					"foo": "v1",
					"bar": "v2",
				},
			},
			new: map[string]interface{}{
				key: map[string]interface{}{
					"foo": "v1",
					"bar": "mutated",
				},
			},
			expectedImmutable:  false,
			expectedAddOnly:    false,
			expectedRemoveOnly: false,
		},
		{
			name: "old not found, new found",
			old:  map[string]interface{}{},
			new: map[string]interface{}{
				key: map[string]interface{}{},
			},
			expectedImmutable:  false,
			expectedAddOnly:    true,
			expectedRemoveOnly: false,
		},
		{
			// foundOld==true, foundNew==false, old[k]=nil (i.e. null in JSON), newObj[k]==nil).
			name: "new not found, old found",
			old: map[string]interface{}{
				key: map[string]interface{}{},
			},
			new:                map[string]interface{}{},
			expectedImmutable:  false,
			expectedAddOnly:    false,
			expectedRemoveOnly: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := isImmutable(structuralschema.XImmutabilityImmutable, tc.old, tc.new, key); got != tc.expectedImmutable {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityImmutable, tc.expectedImmutable, got)
			}
			if got := isImmutable(structuralschema.XImmutabilityAddOnly, tc.old, tc.new, key); got != tc.expectedAddOnly {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityAddOnly, tc.expectedAddOnly, got)
			}
			if got := isImmutable(structuralschema.XImmutabilityRemoveOnly, tc.old, tc.new, key); got != tc.expectedRemoveOnly {
				t.Errorf("%v expected %v, got: %v", structuralschema.XImmutabilityRemoveOnly, tc.expectedRemoveOnly, got)
			}
		})
	}
}
