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

package patch

import (
	"reflect"
	"strings"
	"testing"

	"k8s.io/cli-runtime/pkg/kustomize/k8sdeps/kunstruct"
	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/resid"
	"sigs.k8s.io/kustomize/pkg/resmap"
	"sigs.k8s.io/kustomize/pkg/resource"
)

var rf = resource.NewFactory(
	kunstruct.NewKunstructuredFactoryImpl())
var deploy = gvk.Gvk{Group: "apps", Version: "v1", Kind: "Deployment"}
var foo = gvk.Gvk{Group: "example.com", Version: "v1", Kind: "Foo"}

func TestOverlayRun(t *testing.T) {
	base := resmap.ResMap{
		resid.NewResId(deploy, "deploy1"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata": map[string]interface{}{
					"name": "deploy1",
				},
				"spec": map[string]interface{}{
					"template": map[string]interface{}{
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"old-label": "old-value",
							},
						},
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"name":  "nginx",
									"image": "nginx",
								},
							},
						},
					},
				},
			}),
	}
	patch := []*resource.Resource{
		rf.FromMap(map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"metadata": map[string]interface{}{
						"labels": map[string]interface{}{
							"another-label": "foo",
						},
					},
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:latest",
								"env": []interface{}{
									map[string]interface{}{
										"name":  "SOMEENV",
										"value": "BAR",
									},
								},
							},
						},
					},
				},
			},
		},
		),
	}
	expected := resmap.ResMap{
		resid.NewResId(deploy, "deploy1"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata": map[string]interface{}{
					"name": "deploy1",
				},
				"spec": map[string]interface{}{
					"template": map[string]interface{}{
						"metadata": map[string]interface{}{
							"labels": map[string]interface{}{
								"old-label":     "old-value",
								"another-label": "foo",
							},
						},
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"name":  "nginx",
									"image": "nginx:latest",
									"env": []interface{}{
										map[string]interface{}{
											"name":  "SOMEENV",
											"value": "BAR",
										},
									},
								},
							},
						},
					},
				},
			}),
	}
	lt, err := NewPatchTransformer(patch, rf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = lt.Transform(base)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(base, expected) {
		err = expected.ErrorIfNotEqual(base)
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestMultiplePatches(t *testing.T) {
	base := resmap.ResMap{
		resid.NewResId(deploy, "deploy1"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata": map[string]interface{}{
					"name": "deploy1",
				},
				"spec": map[string]interface{}{
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"name":  "nginx",
									"image": "nginx",
								},
							},
						},
					},
				},
			}),
	}
	patch := []*resource.Resource{
		rf.FromMap(map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:latest",
								"env": []interface{}{
									map[string]interface{}{
										"name":  "SOMEENV",
										"value": "BAR",
									},
								},
							},
						},
					},
				},
			},
		},
		),
		rf.FromMap(map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name": "nginx",
								"env": []interface{}{
									map[string]interface{}{
										"name":  "ANOTHERENV",
										"value": "HELLO",
									},
								},
							},
							map[string]interface{}{
								"name":  "busybox",
								"image": "busybox",
							},
						},
					},
				},
			},
		},
		),
	}
	expected := resmap.ResMap{
		resid.NewResId(deploy, "deploy1"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata": map[string]interface{}{
					"name": "deploy1",
				},
				"spec": map[string]interface{}{
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"name":  "nginx",
									"image": "nginx:latest",
									"env": []interface{}{
										map[string]interface{}{
											"name":  "ANOTHERENV",
											"value": "HELLO",
										},
										map[string]interface{}{
											"name":  "SOMEENV",
											"value": "BAR",
										},
									},
								},
								map[string]interface{}{
									"name":  "busybox",
									"image": "busybox",
								},
							},
						},
					},
				},
			}),
	}
	lt, err := NewPatchTransformer(patch, rf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = lt.Transform(base)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !reflect.DeepEqual(base, expected) {
		err = expected.ErrorIfNotEqual(base)
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestMultiplePatchesWithConflict(t *testing.T) {
	base := resmap.ResMap{
		resid.NewResId(deploy, "deploy1"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "apps/v1",
				"kind":       "Deployment",
				"metadata": map[string]interface{}{
					"name": "deploy1",
				},
				"spec": map[string]interface{}{
					"template": map[string]interface{}{
						"spec": map[string]interface{}{
							"containers": []interface{}{
								map[string]interface{}{
									"name":  "nginx",
									"image": "nginx",
								},
							},
						},
					},
				},
			}),
	}
	patch := []*resource.Resource{
		rf.FromMap(map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:latest",
								"env": []interface{}{
									map[string]interface{}{
										"name":  "SOMEENV",
										"value": "BAR",
									},
								},
							},
						},
					},
				},
			},
		},
		),
		rf.FromMap(map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
			"metadata": map[string]interface{}{
				"name": "deploy1",
			},
			"spec": map[string]interface{}{
				"template": map[string]interface{}{
					"spec": map[string]interface{}{
						"containers": []interface{}{
							map[string]interface{}{
								"name":  "nginx",
								"image": "nginx:1.7.9",
							},
						},
					},
				},
			},
		},
		),
	}

	lt, err := NewPatchTransformer(patch, rf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = lt.Transform(base)
	if err == nil {
		t.Fatalf("did not get expected error")
	}
	if !strings.Contains(err.Error(), "conflict") {
		t.Fatalf("expected error to contain %q but get %v", "conflict", err)
	}
}

func TestNoSchemaOverlayRun(t *testing.T) {
	base := resmap.ResMap{
		resid.NewResId(foo, "my-foo"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "example.com/v1",
				"kind":       "Foo",
				"metadata": map[string]interface{}{
					"name": "my-foo",
				},
				"spec": map[string]interface{}{
					"bar": map[string]interface{}{
						"A": "X",
						"B": "Y",
					},
				},
			}),
	}
	patch := []*resource.Resource{
		rf.FromMap(map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"name": "my-foo",
			},
			"spec": map[string]interface{}{
				"bar": map[string]interface{}{
					"B": nil,
					"C": "Z",
				},
			},
		},
		),
	}
	expected := resmap.ResMap{
		resid.NewResId(foo, "my-foo"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "example.com/v1",
				"kind":       "Foo",
				"metadata": map[string]interface{}{
					"name": "my-foo",
				},
				"spec": map[string]interface{}{
					"bar": map[string]interface{}{
						"A": "X",
						"C": "Z",
					},
				},
			}),
	}

	lt, err := NewPatchTransformer(patch, rf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = lt.Transform(base)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = expected.ErrorIfNotEqual(base); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestNoSchemaMultiplePatches(t *testing.T) {
	base := resmap.ResMap{
		resid.NewResId(foo, "my-foo"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "example.com/v1",
				"kind":       "Foo",
				"metadata": map[string]interface{}{
					"name": "my-foo",
				},
				"spec": map[string]interface{}{
					"bar": map[string]interface{}{
						"A": "X",
						"B": "Y",
					},
				},
			}),
	}
	patch := []*resource.Resource{
		rf.FromMap(map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"name": "my-foo",
			},
			"spec": map[string]interface{}{
				"bar": map[string]interface{}{
					"B": nil,
					"C": "Z",
				},
			},
		},
		),
		rf.FromMap(map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"name": "my-foo",
			},
			"spec": map[string]interface{}{
				"bar": map[string]interface{}{
					"C": "Z",
					"D": "W",
				},
				"baz": map[string]interface{}{
					"hello": "world",
				},
			},
		},
		),
	}
	expected := resmap.ResMap{
		resid.NewResId(foo, "my-foo"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "example.com/v1",
				"kind":       "Foo",
				"metadata": map[string]interface{}{
					"name": "my-foo",
				},
				"spec": map[string]interface{}{
					"bar": map[string]interface{}{
						"A": "X",
						"C": "Z",
						"D": "W",
					},
					"baz": map[string]interface{}{
						"hello": "world",
					},
				},
			}),
	}

	lt, err := NewPatchTransformer(patch, rf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = lt.Transform(base)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if err = expected.ErrorIfNotEqual(base); err != nil {
		t.Fatalf("actual doesn't match expected: %v", err)
	}
}

func TestNoSchemaMultiplePatchesWithConflict(t *testing.T) {
	base := resmap.ResMap{
		resid.NewResId(foo, "my-foo"): rf.FromMap(
			map[string]interface{}{
				"apiVersion": "example.com/v1",
				"kind":       "Foo",
				"metadata": map[string]interface{}{
					"name": "my-foo",
				},
				"spec": map[string]interface{}{
					"bar": map[string]interface{}{
						"A": "X",
						"B": "Y",
					},
				},
			}),
	}
	patch := []*resource.Resource{
		rf.FromMap(map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"name": "my-foo",
			},
			"spec": map[string]interface{}{
				"bar": map[string]interface{}{
					"B": nil,
					"C": "Z",
				},
			},
		}),
		rf.FromMap(map[string]interface{}{
			"apiVersion": "example.com/v1",
			"kind":       "Foo",
			"metadata": map[string]interface{}{
				"name": "my-foo",
			},
			"spec": map[string]interface{}{
				"bar": map[string]interface{}{
					"C": "NOT_Z",
				},
			},
		}),
	}

	lt, err := NewPatchTransformer(patch, rf)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	err = lt.Transform(base)
	if err == nil {
		t.Fatalf("did not get expected error")
	}
	if !strings.Contains(err.Error(), "conflict") {
		t.Fatalf("expected error to contain %q but get %v", "conflict", err)
	}
}
