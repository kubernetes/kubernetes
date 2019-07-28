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

package fieldmanager_test

import (
	"errors"
	"fmt"
	"net/http"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"sigs.k8s.io/yaml"
)

type fakeObjectConvertor struct{}

func (c *fakeObjectConvertor) Convert(in, out, context interface{}) error {
	out = in
	return nil
}

func (c *fakeObjectConvertor) ConvertToVersion(in runtime.Object, _ runtime.GroupVersioner) (runtime.Object, error) {
	return in, nil
}

func (c *fakeObjectConvertor) ConvertFieldLabel(_ schema.GroupVersionKind, _, _ string) (string, string, error) {
	return "", "", errors.New("not implemented")
}

type fakeObjectDefaulter struct{}

func (d *fakeObjectDefaulter) Default(in runtime.Object) {}

func NewTestFieldManager() *fieldmanager.FieldManager {
	gv := schema.GroupVersion{
		Group:   "apps",
		Version: "v1",
	}

	return fieldmanager.NewCRDFieldManager(
		&fakeObjectConvertor{},
		&fakeObjectDefaulter{},
		gv,
		gv,
	)
}

func TestFieldManagerCreation(t *testing.T) {
	if NewTestFieldManager() == nil {
		t.Fatal("failed to create FieldManager")
	}
}

func TestApplyStripsFields(t *testing.T) {
	f := NewTestFieldManager()

	obj := &corev1.Pod{}

	newObj, err := f.Apply(obj, []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "b",
			"namespace": "b",
			"creationTimestamp": "2016-05-19T09:59:00Z",
			"selfLink": "b",
			"uid": "b",
			"clusterName": "b",
			"generation": 0,
			"managedFields": [{
					"manager": "apply",
					"operation": "Apply",
					"apiVersion": "apps/v1",
					"fields": {
						"f:metadata": {
							"f:labels": {
								"f:test-label": {}
							}
						}
					}
				}],
			"resourceVersion": "b"
		}
	}`), "fieldmanager_test", false)
	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	accessor, err := meta.Accessor(newObj)
	if err != nil {
		t.Fatalf("couldn't get accessor: %v", err)
	}

	if m := accessor.GetManagedFields(); len(m) != 0 {
		t.Fatalf("fields did not get stripped on apply: %v", m)
	}
}

func TestVersionCheck(t *testing.T) {
	f := NewTestFieldManager()

	obj := &corev1.Pod{}

	// patch has 'apiVersion: apps/v1' and live version is apps/v1 -> no errors
	_, err := f.Apply(obj, []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
	}`), "fieldmanager_test", false)
	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	// patch has 'apiVersion: apps/v2' but live version is apps/v1 -> error
	_, err = f.Apply(obj, []byte(`{
		"apiVersion": "apps/v2",
		"kind": "Deployment",
	}`), "fieldmanager_test", false)
	if err == nil {
		t.Fatalf("expected an error from mismatched patch and live versions")
	}
	switch typ := err.(type) {
	default:
		t.Fatalf("expected error to be of type %T was %T", apierrors.StatusError{}, typ)
	case apierrors.APIStatus:
		if typ.Status().Code != http.StatusBadRequest {
			t.Fatalf("expected status code to be %d but was %d",
				http.StatusBadRequest, typ.Status().Code)
		}
	}
}

func TestApplyDoesNotStripLabels(t *testing.T) {
	f := NewTestFieldManager()

	obj := &corev1.Pod{}

	newObj, err := f.Apply(obj, []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
		}
	}`), "fieldmanager_test", false)
	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	accessor, err := meta.Accessor(newObj)
	if err != nil {
		t.Fatalf("couldn't get accessor: %v", err)
	}

	if m := accessor.GetManagedFields(); len(m) != 1 {
		t.Fatalf("labels shouldn't get stripped on apply: %v", m)
	}
}

func BenchmarkApplyNewObject(b *testing.B) {
	f := NewTestFieldManager()

	obj := &corev1.Pod{}

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_, err := f.Apply(obj, []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Pod",
		"metadata": {
			"name": "b",
			"namespace": "b",
			"creationTimestamp": "2016-05-19T09:59:00Z",
		},
                "map": {
                        "fieldA": 1,
                        "fieldB": 1,
                        "fieldC": 1,
                        "fieldD": 1,
                        "fieldE": 1,
                        "fieldF": 1,
                        "fieldG": 1,
                        "fieldH": 1,
                        "fieldI": 1,
                        "fieldJ": 1,
                        "fieldK": 1,
                        "fieldL": 1,
                        "fieldM": 1,
                        "fieldN": {
	                        "fieldN": {
					"fieldN": {
						"fieldN": {
				                        "fieldN": {
								"value": true
							},
						},
					},
				},
			},
                }
	}`), "fieldmanager_test", false)
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkUpdateNewObject(b *testing.B) {
	f := NewTestFieldManager()

	oldObj := &corev1.Pod{}
	y := `{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "b",
			"namespace": "b",
			"creationTimestamp": "2016-05-19T09:59:00Z",
		},
                "map": {
                        "fieldA": 1,
                        "fieldB": 1,
                        "fieldC": 1,
                        "fieldD": 1,
                        "fieldE": 1,
                        "fieldF": 1,
                        "fieldG": 1,
                        "fieldH": 1,
                        "fieldI": 1,
                        "fieldJ": 1,
                        "fieldK": 1,
                        "fieldL": 1,
                        "fieldM": 1,
                        "fieldN": {
	                        "fieldN": {
					"fieldN": {
						"fieldN": {
				                        "fieldN": {
								"value": true
							},
						},
					},
				},
			},
		},

	}`
	newObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(y), &newObj.Object); err != nil {
		b.Fatalf("Failed to parse yaml object: %v", err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		_, err := f.Update(oldObj, newObj, "fieldmanager_test")
		if err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkRepeatedUpdate(b *testing.B) {
	f := NewTestFieldManager()

	var oldObj runtime.Object
	oldObj = &unstructured.Unstructured{Object: map[string]interface{}{}}
	y1 := `{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "b",
			"namespace": "b",
			"creationTimestamp": "2016-05-19T09:59:00Z",
		},
                "map": {
                        "fieldA": 1,
                        "fieldB": 1,
                        "fieldC": 1,
                        "fieldD": 1,
                        "fieldE": 1,
                        "fieldF": 1,
                        "fieldG": 1,
                        "fieldH": 1,
                        "fieldI": 1,
                        "fieldJ": 1,
                        "fieldK": 1,
                        "fieldL": 1,
                        "fieldM": 1,
                        "fieldN": {
	                        "fieldN": {
					"fieldN": {
						"fieldN": {
				                        "fieldN": {
								"value": true
							},
						},
					},
				},
			},
		},

	}`
	obj1 := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(y1), &obj1.Object); err != nil {
		b.Fatalf("Failed to parse yaml object: %v", err)
	}
	y2 := `{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "b",
			"namespace": "b",
			"creationTimestamp": "2016-05-19T09:59:00Z",
		},
                "map": {
                        "fieldA": 1,
                        "fieldB": 1,
                        "fieldC": 1,
                        "fieldD": 1,
                        "fieldE": 1,
                        "fieldF": 1,
                        "fieldG": 1,
                        "fieldH": 1,
                        "fieldI": 1,
                        "fieldJ": 1,
                        "fieldK": 1,
                        "fieldL": 1,
                        "fieldM": 1,
                        "fieldN": {
	                        "fieldN": {
					"fieldN": {
						"fieldN": {
				                        "fieldN": {
								"value": false
							},
						},
					},
				},
			},
		},

	}`
	obj2 := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(y2), &obj2.Object); err != nil {
		b.Fatalf("Failed to parse yaml object: %v", err)
	}
	y3 := `{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "b",
			"namespace": "b",
			"creationTimestamp": "2016-05-19T09:59:00Z",
		},
                "map": {
                        "fieldA": 1,
                        "fieldB": 1,
                        "fieldC": 1,
                        "fieldD": 1,
                        "fieldE": 1,
                        "fieldF": 1,
                        "fieldG": 1,
                        "fieldH": 1,
                        "fieldI": 1,
                        "fieldJ": 1,
                        "fieldK": 1,
                        "fieldL": 1,
                        "fieldM": 1,
                        "fieldN": {
	                        "fieldN": {
					"fieldN": {
						"fieldN": {
				                        "fieldN": {
								"value": true
							},
						},
					},
				},
			},
                        "fieldO": 1,
                        "fieldP": 1,
                        "fieldQ": 1,
                        "fieldR": 1,
                        "fieldS": 1,
		},

	}`
	obj3 := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(y3), &obj3.Object); err != nil {
		b.Fatalf("Failed to parse yaml object: %v", err)
	}

	objs := []*unstructured.Unstructured{obj1, obj2, obj3}

	var err error
	oldObj, err = f.Update(oldObj, objs[0], "fieldmanager_0")
	if err != nil {
		b.Fatal(err)
	}

	oldObj, err = f.Update(oldObj, objs[1], "fieldmanager_1")
	if err != nil {
		b.Fatal(err)
	}

	oldObj, err = f.Update(oldObj, objs[2], "fieldmanager_2")
	if err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		oldObj, err = f.Update(oldObj, objs[n%3], fmt.Sprintf("fieldmanager_%d", n%3))
		if err != nil {
			b.Fatal(err)
		}
	}
}
