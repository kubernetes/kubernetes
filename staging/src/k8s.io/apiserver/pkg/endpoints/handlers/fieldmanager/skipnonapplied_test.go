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
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"sigs.k8s.io/yaml"
)

type fakeObjectCreater struct {
	gvk schema.GroupVersionKind
}

var _ runtime.ObjectCreater = &fakeObjectCreater{}

func (f *fakeObjectCreater) New(_ schema.GroupVersionKind) (runtime.Object, error) {
	u := unstructured.Unstructured{Object: map[string]interface{}{}}
	u.SetAPIVersion(f.gvk.GroupVersion().String())
	u.SetKind(f.gvk.Kind)
	return &u, nil
}

func TestNoUpdateBeforeFirstApply(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))
	f.fieldManager = fieldmanager.NewSkipNonAppliedManager(
		f.fieldManager,
		&fakeObjectCreater{gvk: schema.GroupVersionKind{Version: "v1", Kind: "Pod"}},
		schema.GroupVersionKind{},
	)

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"name": "pod",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"containers": [{
				"name":  "nginx",
				"image": "nginx:latest"
			}]
        }
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	if err := f.Apply(appliedObj, "fieldmanager_test_apply", false); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}

	if e, a := 1, len(f.ManagedFields()); e != a {
		t.Fatalf("exected %v entries in managedFields, but got %v: %#v", e, a, f.ManagedFields())
	}

	if e, a := "fieldmanager_test_apply", f.ManagedFields()[0].Manager; e != a {
		t.Fatalf("exected manager name to be %v, but got %v: %#v", e, a, f.ManagedFields())
	}
}

func TestUpdateBeforeFirstApply(t *testing.T) {
	f := NewTestFieldManager(schema.FromAPIVersionAndKind("v1", "Pod"))
	f.fieldManager = fieldmanager.NewSkipNonAppliedManager(
		f.fieldManager,
		&fakeObjectCreater{gvk: schema.GroupVersionKind{Version: "v1", Kind: "Pod"}},
		schema.GroupVersionKind{},
	)

	updatedObj := &corev1.Pod{}
	updatedObj.Kind = "Pod"
	updatedObj.APIVersion = "v1"
	updatedObj.ObjectMeta.Labels = map[string]string{"app": "my-nginx"}

	if err := f.Update(updatedObj, "fieldmanager_test_update"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}

	if m := f.ManagedFields(); len(m) != 0 {
		t.Fatalf("managedFields were tracked on update only: %v", m)
	}

	appliedObj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"name": "pod",
			"labels": {"app": "nginx"}
		},
		"spec": {
			"containers": [{
				"name":  "nginx",
				"image": "nginx:latest"
			}]
        }
	}`), &appliedObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	err := f.Apply(appliedObj, "fieldmanager_test_apply", false)
	apiStatus, _ := err.(apierrors.APIStatus)
	if err == nil || !apierrors.IsConflict(err) || len(apiStatus.Status().Details.Causes) != 1 {
		t.Fatalf("Expecting to get one conflict but got %v", err)
	}

	if e, a := ".metadata.labels.app", apiStatus.Status().Details.Causes[0].Field; e != a {
		t.Fatalf("Expecting to conflict on field %q but conflicted on field %q: %v", e, a, err)
	}

	if e, a := "before-first-apply", apiStatus.Status().Details.Causes[0].Message; !strings.Contains(a, e) {
		t.Fatalf("Expecting conflict message to contain %q but got %q: %v", e, a, err)
	}

	if err := f.Apply(appliedObj, "fieldmanager_test_apply", true); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}

	if e, a := 2, len(f.ManagedFields()); e != a {
		t.Fatalf("exected %v entries in managedFields, but got %v: %#v", e, a, f.ManagedFields())
	}

	if e, a := "fieldmanager_test_apply", f.ManagedFields()[0].Manager; e != a {
		t.Fatalf("exected first manager name to be %v, but got %v: %#v", e, a, f.ManagedFields())
	}

	if e, a := "before-first-apply", f.ManagedFields()[1].Manager; e != a {
		t.Fatalf("exected second manager name to be %v, but got %v: %#v", e, a, f.ManagedFields())
	}
}
