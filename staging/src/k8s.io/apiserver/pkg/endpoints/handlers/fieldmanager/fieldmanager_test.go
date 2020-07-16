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
	"reflect"
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
	"k8s.io/apimachinery/pkg/runtime/serializer"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	"k8s.io/kube-openapi/pkg/util/proto"
	prototesting "k8s.io/kube-openapi/pkg/util/proto/testing"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
	"sigs.k8s.io/structured-merge-diff/v4/typed"
	"sigs.k8s.io/yaml"
)

var fakeSchema = prototesting.Fake{
	Path: filepath.Join(
		strings.Repeat(".."+string(filepath.Separator), 8),
		"api", "openapi-spec", "swagger.json"),
}

type fakeObjectConvertor struct {
	converter  merge.Converter
	apiVersion fieldpath.APIVersion
}

func (c *fakeObjectConvertor) Convert(in, out, context interface{}) error {
	if typedValue, ok := in.(*typed.TypedValue); ok {
		var err error
		out, err = c.converter.Convert(typedValue, c.apiVersion)
		return err
	}
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
	fieldManager *fieldmanager.FieldManager
	emptyObj     runtime.Object
	liveObj      runtime.Object
}

func NewDefaultTestFieldManager(gvk schema.GroupVersionKind) TestFieldManager {
	return NewTestFieldManager(gvk, nil)
}

func NewTestFieldManager(gvk schema.GroupVersionKind, chainFieldManager func(fieldmanager.Manager) fieldmanager.Manager) TestFieldManager {
	m := NewFakeOpenAPIModels()
	typeConverter := NewFakeTypeConverter(m)
	converter := internal.NewVersionConverter(typeConverter, &fakeObjectConvertor{}, gvk.GroupVersion())
	apiVersion := fieldpath.APIVersion(gvk.GroupVersion().String())
	objectConverter := &fakeObjectConvertor{converter, apiVersion}
	f, err := fieldmanager.NewStructuredMergeManager(
		typeConverter,
		objectConverter,
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
	f = fieldmanager.NewManagedFieldsUpdater(f)
	f = fieldmanager.NewBuildManagerInfoManager(f, gvk.GroupVersion())
	f = fieldmanager.NewProbabilisticSkipNonAppliedManager(f, &fakeObjectCreater{gvk: gvk}, gvk, fieldmanager.DefaultTrackOnCreateProbability)
	f = fieldmanager.NewLastAppliedManager(f, typeConverter, objectConverter, gvk.GroupVersion())
	f = fieldmanager.NewLastAppliedUpdater(f)
	if chainFieldManager != nil {
		f = chainFieldManager(f)
	}
	return TestFieldManager{
		fieldManager: fieldmanager.NewFieldManager(f),
		emptyObj:     live,
		liveObj:      live.DeepCopyObject(),
	}
}

func NewFakeTypeConverter(m proto.Models) internal.TypeConverter {
	tc, err := internal.NewTypeConverter(m, false)
	if err != nil {
		panic(fmt.Sprintf("Failed to build TypeConverter: %v", err))
	}
	return tc
}

func NewFakeOpenAPIModels() proto.Models {
	d, err := fakeSchema.OpenAPISchema()
	if err != nil {
		panic(err)
	}
	m, err := proto.NewOpenAPIData(d)
	if err != nil {
		panic(err)
	}
	return m
}

func (f *TestFieldManager) Reset() {
	f.liveObj = f.emptyObj.DeepCopyObject()
}

func (f *TestFieldManager) Apply(obj runtime.Object, manager string, force bool) error {
	out, err := f.fieldManager.Apply(f.liveObj, obj, manager, force)
	if err == nil {
		f.liveObj = out
	}
	return err
}

func (f *TestFieldManager) Update(obj runtime.Object, manager string) error {
	out, err := f.fieldManager.Update(f.liveObj, obj, manager)
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
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

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

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
		},
		"spec": {
			"replicas": 101,
		}
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	err := f.Apply(appliedObj, "fieldmanager_conflict", false)
	if err == nil || !apierrors.IsConflict(err) {
		t.Fatalf("Expecting to get conflicts but got %v", err)
	}
}

func TestApplyStripsFields(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

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
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	// patch has 'apiVersion: apps/v1' and live version is apps/v1 -> no errors
	err := f.Apply(appliedObj, "fieldmanager_test", false)
	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	appliedObj = &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "apps/v1beta1",
		"kind": "Deployment",
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	// patch has 'apiVersion: apps/v2' but live version is apps/v1 -> error
	err = f.Apply(appliedObj, "fieldmanager_test", false)
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
func TestVersionCheckDoesNotPanic(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	// patch has 'apiVersion: apps/v1' and live version is apps/v1 -> no errors
	err := f.Apply(appliedObj, "fieldmanager_test", false)
	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	appliedObj = &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	// patch has 'apiVersion: apps/v2' but live version is apps/v1 -> error
	err = f.Apply(appliedObj, "fieldmanager_test", false)
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
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
		}
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	err := f.Apply(appliedObj, "fieldmanager_test", false)
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
			f := NewDefaultTestFieldManager(test.gvk)

			appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
			if err := yaml.Unmarshal(test.obj, &appliedObj.Object); err != nil {
				t.Fatalf("error decoding YAML: %v", err)
			}

			if err := f.Apply(appliedObj, "fieldmanager_test", false); err != nil {
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
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		b.Fatalf("Failed to add to scheme: %v", err)
	}
	for _, test := range tests {
		b.Run(test.gvk.Kind, func(b *testing.B) {
			f := NewDefaultTestFieldManager(test.gvk)

			decoder := serializer.NewCodecFactory(scheme).UniversalDecoder(test.gvk.GroupVersion())
			newObj, err := runtime.Decode(decoder, test.obj)
			if err != nil {
				b.Fatalf("Failed to parse yaml object: %v", err)
			}
			objMeta, err := meta.Accessor(newObj)
			if err != nil {
				b.Fatalf("Failed to get object meta: %v", err)
			}
			objMeta.SetManagedFields([]metav1.ManagedFieldsEntry{
				{
					Manager:    "default",
					Operation:  "Update",
					APIVersion: "v1",
				},
			})
			appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
			if err := yaml.Unmarshal(test.obj, &appliedObj.Object); err != nil {
				b.Fatalf("Failed to parse yaml object: %v", err)
			}
			b.Run("Update", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					if err := f.Update(newObj, "fieldmanager_test"); err != nil {
						b.Fatal(err)
					}
					f.Reset()
				}
			})
			b.Run("UpdateTwice", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					if err := f.Update(newObj, "fieldmanager_test"); err != nil {
						b.Fatal(err)
					}
					if err := f.Update(newObj, "fieldmanager_test_2"); err != nil {
						b.Fatal(err)
					}
					f.Reset()
				}
			})
			b.Run("Apply", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					if err := f.Apply(appliedObj, "fieldmanager_test", false); err != nil {
						b.Fatal(err)
					}
					f.Reset()
				}
			})
			b.Run("UpdateApply", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for n := 0; n < b.N; n++ {
					if err := f.Update(newObj, "fieldmanager_test"); err != nil {
						b.Fatal(err)
					}
					if err := f.Apply(appliedObj, "fieldmanager_test", false); err != nil {
						b.Fatal(err)
					}
					f.Reset()
				}
			})
		})
	}
}

func toUnstructured(b *testing.B, o runtime.Object) *unstructured.Unstructured {
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(o)
	if err != nil {
		b.Fatalf("Failed to unmarshal to json: %v", err)
	}
	return &unstructured.Unstructured{Object: u}
}

func BenchmarkConvertObjectToTyped(b *testing.B) {
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
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		b.Fatalf("Failed to add to scheme: %v", err)
	}

	for _, test := range tests {
		b.Run(test.gvk.Kind, func(b *testing.B) {
			decoder := serializer.NewCodecFactory(scheme).UniversalDecoder(test.gvk.GroupVersion())
			m := NewFakeOpenAPIModels()
			typeConverter := NewFakeTypeConverter(m)

			structured, err := runtime.Decode(decoder, test.obj)
			if err != nil {
				b.Fatalf("Failed to parse yaml object: %v", err)
			}
			b.Run("structured", func(b *testing.B) {
				b.ReportAllocs()
				b.RunParallel(func(pb *testing.PB) {
					for pb.Next() {
						_, err := typeConverter.ObjectToTyped(structured)
						if err != nil {
							b.Errorf("Error in ObjectToTyped: %v", err)
						}
					}
				})
			})

			unstructured := toUnstructured(b, structured)
			b.Run("unstructured", func(b *testing.B) {
				b.ReportAllocs()
				b.RunParallel(func(pb *testing.PB) {
					for pb.Next() {
						_, err := typeConverter.ObjectToTyped(unstructured)
						if err != nil {
							b.Errorf("Error in ObjectToTyped: %v", err)
						}
					}
				})
			})
		})
	}
}

func BenchmarkCompare(b *testing.B) {
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

	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		b.Fatalf("Failed to add to scheme: %v", err)
	}

	for _, test := range tests {
		b.Run(test.gvk.Kind, func(b *testing.B) {
			decoder := serializer.NewCodecFactory(scheme).UniversalDecoder(test.gvk.GroupVersion())
			m := NewFakeOpenAPIModels()
			typeConverter := NewFakeTypeConverter(m)

			structured, err := runtime.Decode(decoder, test.obj)
			if err != nil {
				b.Fatal(err)
			}
			tv1, err := typeConverter.ObjectToTyped(structured)
			if err != nil {
				b.Errorf("Error in ObjectToTyped: %v", err)
			}
			tv2, err := typeConverter.ObjectToTyped(structured)
			if err != nil {
				b.Errorf("Error in ObjectToTyped: %v", err)
			}

			b.Run("structured", func(b *testing.B) {
				b.ReportAllocs()
				for n := 0; n < b.N; n++ {
					_, err = tv1.Compare(tv2)
					if err != nil {
						b.Errorf("Error in ObjectToTyped: %v", err)
					}
				}
			})

			unstructured := toUnstructured(b, structured)
			utv1, err := typeConverter.ObjectToTyped(unstructured)
			if err != nil {
				b.Errorf("Error in ObjectToTyped: %v", err)
			}
			utv2, err := typeConverter.ObjectToTyped(unstructured)
			if err != nil {
				b.Errorf("Error in ObjectToTyped: %v", err)
			}
			b.Run("unstructured", func(b *testing.B) {
				b.ReportAllocs()
				b.RunParallel(func(pb *testing.PB) {
					for pb.Next() {
						_, err = utv1.Compare(utv2)
						if err != nil {
							b.Errorf("Error in ObjectToTyped: %v", err)
						}
					}
				})
			})
		})
	}
}

func BenchmarkRepeatedUpdate(b *testing.B) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))
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

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal(podBytes, &appliedObj.Object); err != nil {
		b.Fatalf("error decoding YAML: %v", err)
	}

	err := f.Apply(appliedObj, "fieldmanager_apply", false)
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
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"managedFields": [
				{
				  "manager": "test",
				}
			]
		}
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	err := f.Apply(appliedObj, "fieldmanager_test", false)

	if err == nil {
		t.Fatalf("successfully applied with set managed fields")
	}
}

func TestApplySuccessWithNoManagedFields(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
		}
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	err := f.Apply(appliedObj, "fieldmanager_test", false)

	if err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}
}

// Run an update and apply, and make sure that nothing has changed.
func TestNoOpChanges(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
		}
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	if err := f.Apply(obj, "fieldmanager_test_apply", false); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}
	before := f.liveObj.DeepCopyObject()
	// Wait to make sure the timestamp is different
	time.Sleep(time.Second)
	// Applying with a different fieldmanager will create an entry..
	if err := f.Apply(obj, "fieldmanager_test_apply_other", false); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	if reflect.DeepEqual(before, f.liveObj) {
		t.Fatalf("Applying no-op apply with new manager didn't change object: \n%v", f.liveObj)
	}
	before = f.liveObj.DeepCopyObject()
	// Wait to make sure the timestamp is different
	time.Sleep(time.Second)
	if err := f.Update(obj, "fieldmanager_test_update"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	if !reflect.DeepEqual(before, f.liveObj) {
		t.Fatalf("No-op update has changed the object:\n%v\n---\n%v", before, f.liveObj)
	}
	before = f.liveObj.DeepCopyObject()
	// Wait to make sure the timestamp is different
	time.Sleep(time.Second)
	if err := f.Apply(obj, "fieldmanager_test_apply", true); err != nil {
		t.Fatalf("failed to re-apply object: %v", err)
	}
	if !reflect.DeepEqual(before, f.liveObj) {
		t.Fatalf("No-op apply has changed the object:\n%v\n---\n%v", before, f.liveObj)
	}
}

// Tests that one can reset the managedFields by sending either an empty
// list
func TestResetManagedFieldsEmptyList(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
		}
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	if err := f.Apply(obj, "fieldmanager_test_apply", false); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"managedFields": [],
			"labels": {
				"a": "b"
			},
		}
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	if err := f.Update(obj, "update_manager"); err != nil {
		t.Fatalf("failed to update with empty manager: %v", err)
	}

	if len(f.ManagedFields()) != 0 {
		t.Fatalf("failed to reset managedFields: %v", f.ManagedFields())
	}
}

// Tests that one can reset the managedFields by sending either a list with one empty item.
func TestResetManagedFieldsEmptyItem(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))

	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
		}
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	if err := f.Apply(obj, "fieldmanager_test_apply", false); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"managedFields": [{}],
			"labels": {
				"a": "b"
			},
		}
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	if err := f.Update(obj, "update_manager"); err != nil {
		t.Fatalf("failed to update with empty manager: %v", err)
	}

	if len(f.ManagedFields()) != 0 {
		t.Fatalf("failed to reset managedFields: %v", f.ManagedFields())
	}
}

func TestServerSideApplyWithInvalidLastApplied(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	// create object with client-side apply
	newObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	deployment := []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app-v1
spec:
  replicas: 1
`)
	if err := yaml.Unmarshal(deployment, &newObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}

	invalidLastApplied := "invalid-object"
	if err := setLastApplied(newObj, invalidLastApplied); err != nil {
		t.Errorf("failed to set last applied: %v", err)
	}

	if err := f.Update(newObj, "kubectl-client-side-apply-test"); err != nil {
		t.Errorf("failed to update object: %v", err)
	}

	lastApplied, err := getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if lastApplied != invalidLastApplied {
		t.Errorf("expected last applied annotation to be set to %q, but got: %q", invalidLastApplied, lastApplied)
	}

	// upgrade management of the object from client-side apply to server-side apply
	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	appliedDeployment := []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app-v2
spec:
  replicas: 100
`)
	if err := yaml.Unmarshal(appliedDeployment, &appliedObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}

	if err := f.Apply(appliedObj, "kubectl", false); err == nil || !apierrors.IsConflict(err) {
		t.Errorf("expected conflict when applying with invalid last-applied annotation, but got no error for object: \n%+v", appliedObj)
	}

	lastApplied, err = getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if lastApplied != invalidLastApplied {
		t.Errorf("expected last applied annotation to be NOT be updated, but got: %q", lastApplied)
	}

	// force server-side apply should work and fix the annotation
	if err := f.Apply(appliedObj, "kubectl", true); err != nil {
		t.Errorf("failed to force server-side apply with: %v", err)
	}

	lastApplied, err = getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if lastApplied == invalidLastApplied ||
		!strings.Contains(lastApplied, "my-app-v2") {
		t.Errorf("expected last applied annotation to be updated, but got: %q", lastApplied)
	}
}

func TestInteropForClientSideApplyAndServerSideApply(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	// create object with client-side apply
	newObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	deployment := []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100
  selector:
    matchLabels:
      app: my-app
  template:
    metadata:
      labels:
        app: my-app
    spec:
      containers:
      - name: my-c
        image: my-image-v1
`)
	if err := yaml.Unmarshal(deployment, &newObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}
	if err := setLastAppliedFromEncoded(newObj, deployment); err != nil {
		t.Errorf("failed to set last applied: %v", err)
	}

	if err := f.Update(newObj, "kubectl-client-side-apply-test"); err != nil {
		t.Errorf("failed to update object: %v", err)
	}
	lastApplied, err := getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if !strings.Contains(lastApplied, "my-image-v1") {
		t.Errorf("expected last applied annotation to be set properly, but got: %q", lastApplied)
	}

	// upgrade management of the object from client-side apply to server-side apply
	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	appliedDeployment := []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app-v2 # change
spec:
  replicas: 8 # change
  selector:
    matchLabels:
      app: my-app-v2 # change
  template:
    metadata:
      labels:
        app: my-app-v2 # change
    spec:
      containers:
      - name: my-c
        image: my-image-v2 # change
`)
	if err := yaml.Unmarshal(appliedDeployment, &appliedObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}

	if err := f.Apply(appliedObj, "kubectl", false); err != nil {
		t.Errorf("error applying object: %v", err)
	}

	lastApplied, err = getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if !strings.Contains(lastApplied, "my-image-v2") {
		t.Errorf("expected last applied annotation to be updated, but got: %q", lastApplied)
	}
}

func TestNoTrackManagedFieldsForClientSideApply(t *testing.T) {
	f := NewDefaultTestFieldManager(schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	// create object
	newObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	deployment := []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100
`)
	if err := yaml.Unmarshal(deployment, &newObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}
	if err := f.Update(newObj, "test_kubectl_create"); err != nil {
		t.Errorf("failed to update object: %v", err)
	}
	if m := f.ManagedFields(); len(m) == 0 {
		t.Errorf("expected to have managed fields, but got: %v", m)
	}

	// stop tracking managed fields
	newObj = &unstructured.Unstructured{Object: map[string]interface{}{}}
	deployment = []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  managedFields: [] # stop tracking managed fields
  labels:
    app: my-app
spec:
  replicas: 100
`)
	if err := yaml.Unmarshal(deployment, &newObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}
	newObj.SetUID("nonempty")
	if err := f.Update(newObj, "test_kubectl_replace"); err != nil {
		t.Errorf("failed to update object: %v", err)
	}
	if m := f.ManagedFields(); len(m) != 0 {
		t.Errorf("expected to have stop tracking managed fields, but got: %v", m)
	}

	// check that we still don't track managed fields
	newObj = &unstructured.Unstructured{Object: map[string]interface{}{}}
	deployment = []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100
`)
	if err := yaml.Unmarshal(deployment, &newObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}
	if err := setLastAppliedFromEncoded(newObj, deployment); err != nil {
		t.Errorf("failed to set last applied: %v", err)
	}
	if err := f.Update(newObj, "test_k_client_side_apply"); err != nil {
		t.Errorf("failed to update object: %v", err)
	}
	if m := f.ManagedFields(); len(m) != 0 {
		t.Errorf("expected to continue to not track managed fields, but got: %v", m)
	}
	lastApplied, err := getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if !strings.Contains(lastApplied, "my-app") {
		t.Errorf("expected last applied annotation to be set properly, but got: %q", lastApplied)
	}

	// start tracking managed fields
	newObj = &unstructured.Unstructured{Object: map[string]interface{}{}}
	deployment = []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app
spec:
  replicas: 100
`)
	if err := yaml.Unmarshal(deployment, &newObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}
	if err := f.Apply(newObj, "test_server_side_apply_without_upgrade", false); err != nil {
		t.Errorf("error applying object: %v", err)
	}
	if m := f.ManagedFields(); len(m) < 2 {
		t.Errorf("expected to start tracking managed fields with at least 2 field managers, but got: %v", m)
	}
	if e, a := "test_server_side_apply_without_upgrade", f.ManagedFields()[0].Manager; e != a {
		t.Fatalf("exected first manager name to be %v, but got %v: %#v", e, a, f.ManagedFields())
	}
	if e, a := "before-first-apply", f.ManagedFields()[1].Manager; e != a {
		t.Fatalf("exected second manager name to be %v, but got %v: %#v", e, a, f.ManagedFields())
	}

	// upgrade management of the object from client-side apply to server-side apply
	newObj = &unstructured.Unstructured{Object: map[string]interface{}{}}
	deployment = []byte(`
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-deployment
  labels:
    app: my-app-v2 # change
spec:
  replicas: 8 # change
`)
	if err := yaml.Unmarshal(deployment, &newObj.Object); err != nil {
		t.Errorf("error decoding YAML: %v", err)
	}
	if err := f.Apply(newObj, "kubectl", false); err != nil {
		t.Errorf("error applying object: %v", err)
	}
	if m := f.ManagedFields(); len(m) == 0 {
		t.Errorf("expected to track managed fields, but got: %v", m)
	}
	lastApplied, err = getLastApplied(f.liveObj)
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if !strings.Contains(lastApplied, "my-app-v2") {
		t.Errorf("expected last applied annotation to be updated, but got: %q", lastApplied)
	}
}

func yamlToJSON(y []byte) (string, error) {
	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal(y, &obj.Object); err != nil {
		return "", fmt.Errorf("error decoding YAML: %v", err)
	}
	serialization, err := runtime.Encode(unstructured.UnstructuredJSONScheme, obj)
	if err != nil {
		return "", fmt.Errorf("error encoding object: %v", err)
	}
	json, err := yamlutil.ToJSON(serialization)
	if err != nil {
		return "", fmt.Errorf("error converting to json: %v", err)
	}
	return string(json), nil
}

func setLastAppliedFromEncoded(obj runtime.Object, lastApplied []byte) error {
	lastAppliedJSON, err := yamlToJSON(lastApplied)
	if err != nil {
		return err
	}
	return setLastApplied(obj, lastAppliedJSON)
}

func setLastApplied(obj runtime.Object, lastApplied string) error {
	accessor := meta.NewAccessor()
	annotations, err := accessor.Annotations(obj)
	if err != nil {
		return fmt.Errorf("failed to access annotations: %v", err)
	}
	if annotations == nil {
		annotations = map[string]string{}
	}
	annotations[corev1.LastAppliedConfigAnnotation] = lastApplied
	accessor.SetAnnotations(obj, annotations)
	return nil
}

func getLastApplied(obj runtime.Object) (string, error) {
	accessor := meta.NewAccessor()
	annotations, err := accessor.Annotations(obj)
	if err != nil {
		return "", fmt.Errorf("failed to access annotations: %v", err)
	}
	if annotations == nil {
		return "", fmt.Errorf("no annotations on obj: %v", obj)
	}

	lastApplied, ok := annotations[corev1.LastAppliedConfigAnnotation]
	if !ok {
		return "", fmt.Errorf("expected last applied annotation, but got none for object: %v", obj)
	}
	return lastApplied, nil
}
