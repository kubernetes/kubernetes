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
	"io/ioutil"
	"net/http"
	"path/filepath"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/kube-openapi/pkg/util/proto"
	prototesting "k8s.io/kube-openapi/pkg/util/proto/testing"
	"sigs.k8s.io/yaml"
)

var fakeSchema = prototesting.Fake{
	Path: filepath.Join(
		strings.Repeat(".."+string(filepath.Separator), 8),
		"api", "openapi-spec", "swagger.json"),
}

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

type TestFieldManager struct {
	fieldManager fieldmanager.Manager
	emptyObj     runtime.Object
	liveObj      runtime.Object
}

func NewTestFieldManager(gvk schema.GroupVersionKind) TestFieldManager {
	d, err := fakeSchema.OpenAPISchema()
	if err != nil {
		panic(err)
	}
	m, err := proto.NewOpenAPIData(d)
	if err != nil {
		panic(err)
	}

	f, err := fieldmanager.NewStructuredMergeManager(
		m,
		&fakeObjectConvertor{},
		&fakeObjectDefaulter{},
		gvk.GroupVersion(),
		gvk.GroupVersion(),
	)
	if err != nil {
		panic(err)
	}
	live := &unstructured.Unstructured{}
	live.SetKind(gvk.Kind)
	live.SetAPIVersion(gvk.GroupVersion().String())
	f = fieldmanager.NewStripMetaManager(f)
	f = fieldmanager.NewBuildManagerInfoManager(f, gvk.GroupVersion())
	return TestFieldManager{
		fieldManager: f,
		emptyObj:     live,
		liveObj:      live.DeepCopyObject(),
	}
}

func (f *TestFieldManager) Reset() {
	f.liveObj = f.emptyObj.DeepCopyObject()
}

func (f *TestFieldManager) Apply(obj []byte, manager string, force bool) error {
	out, err := fieldmanager.NewFieldManager(f.fieldManager).Apply(f.liveObj, obj, manager, force)
	if err == nil {
		f.liveObj = out
	}
	return err
}

func (f *TestFieldManager) Update(obj runtime.Object, manager string) error {
	out, err := fieldmanager.NewFieldManager(f.fieldManager).Update(f.liveObj, obj, manager)
	if err == nil {
		f.liveObj = out
	}
	return err
}

func (f *TestFieldManager) ManagedFields() []metav1.ManagedFieldsEntry {
	accessor, err := meta.Accessor(f.liveObj)
	if err != nil {
		panic(fmt.Errorf("couldn't get accessor: %v", err))
	}

	return accessor.GetManagedFields()
}

// TestUpdateApplyConflict tests that applying to an object, which
// wasn't created by apply, will give conflicts
func TestUpdateApplyConflict(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	patch := []byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
                        "replicas": 3,
                        "selector": {
                                "matchLabels": {
                                         "app": "nginx"
                                }
                        },
                        "template": {
                                "metadata": {
                                        "labels": {
                                                "app": "nginx"
                                        }
                                },
                                "spec": {
				        "containers": [{
					        "name":  "nginx",
					        "image": "nginx:latest"
				        }]
                                }
                        }
		}
	}`)
	newObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal(patch, &newObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	if err := f.Update(newObj, "fieldmanager_test"); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	err := f.Apply([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
		},
		"spec": {
			"replicas": 101,
		}
	}`), "fieldmanager_conflict", false)
	if err == nil || !apierrors.IsConflict(err) {
		t.Fatalf("Expecting to get conflicts but got %v", err)
	}
}

func TestApplyStripsFields(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	newObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
		},
	}

	newObj.SetName("b")
	newObj.SetNamespace("b")
	newObj.SetUID("b")
	newObj.SetClusterName("b")
	newObj.SetGeneration(0)
	newObj.SetResourceVersion("b")
	newObj.SetCreationTimestamp(metav1.NewTime(time.Now()))
	newObj.SetManagedFields([]metav1.ManagedFieldsEntry{
		{
			Manager:    "update",
			Operation:  metav1.ManagedFieldsOperationApply,
			APIVersion: "apps/v1",
		},
	})
	if err := f.Update(newObj, "fieldmanager_test"); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	if m := f.ManagedFields(); len(m) != 0 {
		t.Fatalf("fields did not get stripped: %v", m)
	}
}

func TestVersionCheck(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	// patch has 'apiVersion: apps/v1' and live version is apps/v1 -> no errors
	err := f.Apply([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
	}`), "fieldmanager_test", false)
	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	// patch has 'apiVersion: apps/v2' but live version is apps/v1 -> error
	err = f.Apply([]byte(`{
		"apiVersion": "apps/v1beta1",
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
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	err := f.Apply([]byte(`{
		"apiVersion": "v1",
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

	if m := f.ManagedFields(); len(m) != 1 {
		t.Fatalf("labels shouldn't get stripped on apply: %v", m)
	}
}

func getObjectBytes(file string) []byte {
	s, err := ioutil.ReadFile(file)
	if err != nil {
		panic(err)
	}
	return s
}

func TestApplyNewObject(t *testing.T) {
	tests := []struct {
		gvk schema.GroupVersionKind
		obj []byte
	}{
		{
			gvk: schema.FromAPIVersionAndKind("v1", "Pod"),
			obj: getObjectBytes("pod.yaml"),
		},
		{
			gvk: schema.FromAPIVersionAndKind("v1", "Node"),
			obj: getObjectBytes("node.yaml"),
		},
		{
			gvk: schema.FromAPIVersionAndKind("v1", "Endpoints"),
			obj: getObjectBytes("endpoints.yaml"),
		},
	}

	for _, test := range tests {
		t.Run(test.gvk.String(), func(t *testing.T) {
			f := NewTestFieldManager(test.gvk)

			if err := f.Apply(test.obj, "fieldmanager_test", false); err != nil {
				t.Fatal(err)
			}
		})
	}
}

func BenchmarkNewObject(b *testing.B) {
	tests := []struct {
		gvk schema.GroupVersionKind
		obj []byte
	}{
		{
			gvk: schema.FromAPIVersionAndKind("v1", "Pod"),
			obj: getObjectBytes("pod.yaml"),
		},
		{
			gvk: schema.FromAPIVersionAndKind("v1", "Node"),
			obj: getObjectBytes("node.yaml"),
		},
		{
			gvk: schema.FromAPIVersionAndKind("v1", "Endpoints"),
			obj: getObjectBytes("endpoints.yaml"),
		},
	}

	for _, test := range tests {
		b.Run(test.gvk.Kind, func(b *testing.B) {
			f := NewTestFieldManager(test.gvk)

			newObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
			if err := yaml.Unmarshal(test.obj, &newObj.Object); err != nil {
				b.Fatalf("Failed to parse yaml object: %v", err)
			}
			newObj.SetManagedFields([]metav1.ManagedFieldsEntry{
				{
					Manager:    "default",
					Operation:  "Update",
					APIVersion: "v1",
				},
			})

			b.Run("Update", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					err := f.Update(newObj, "fieldmanager_test")
					if err != nil {
						b.Fatal(err)
					}
					f.Reset()
				}
			})
			b.Run("Apply", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					err := f.Apply(test.obj, "fieldmanager_test", false)
					if err != nil {
						b.Fatal(err)
					}
					f.Reset()
				}
			})
		})
	}
}

func BenchmarkRepeatedUpdate(b *testing.B) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))
	podBytes := getObjectBytes("pod.yaml")

	var obj *corev1.Pod
	if err := yaml.Unmarshal(podBytes, &obj); err != nil {
		b.Fatalf("Failed to parse yaml object: %v", err)
	}
	obj.Spec.Containers[0].Image = "nginx:latest"
	objs := []*corev1.Pod{obj}
	obj = obj.DeepCopy()
	obj.Spec.Containers[0].Image = "nginx:4.3"
	objs = append(objs, obj)

	err := f.Apply(podBytes, "fieldmanager_apply", false)
	if err != nil {
		b.Fatal(err)
	}

	if err := f.Update(objs[1], "fieldmanager_1"); err != nil {
		b.Fatal(err)
	}

	b.ReportAllocs()
	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		err := f.Update(objs[n%len(objs)], fmt.Sprintf("fieldmanager_%d", n%len(objs)))
		if err != nil {
			b.Fatal(err)
		}
		f.Reset()
	}
}

func TestApplyFailsWithManagedFields(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	err := f.Apply([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"managedFields": [
				{
				  "manager": "test",
				}
			]
		}
	}`), "fieldmanager_test", false)

	if err == nil {
		t.Fatalf("successfully applied with set managed fields")
	}
}

func TestApplySuccessWithNoManagedFields(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	err := f.Apply([]byte(`{
		"apiVersion": "v1",
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
}
