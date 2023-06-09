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

package managedfields_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/managedfields"
	"k8s.io/apimachinery/pkg/util/managedfields/internal"
	"k8s.io/apimachinery/pkg/util/managedfields/managedfieldstest"
	yamlutil "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/kube-openapi/pkg/validation/spec"
	"sigs.k8s.io/yaml"
)

var fakeTypeConverter = func() managedfields.TypeConverter {
	data, err := os.ReadFile(filepath.Join(strings.Repeat(".."+string(filepath.Separator), 7),
		"api", "openapi-spec", "swagger.json"))
	if err != nil {
		panic(err)
	}
	swag := spec.Swagger{}
	if err := json.Unmarshal(data, &swag); err != nil {
		panic(err)
	}
	convertedDefs := map[string]*spec.Schema{}
	for k, v := range swag.Definitions {
		vCopy := v
		convertedDefs[k] = &vCopy
	}
	typeConverter, err := managedfields.NewTypeConverter(convertedDefs, false)
	if err != nil {
		panic(err)
	}
	return typeConverter
}()

// TestUpdateApplyConflict tests that applying to an object, which
// wasn't created by apply, will give conflicts
func TestUpdateApplyConflict(t *testing.T) {
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

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
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

	newObj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apps/v1",
			"kind":       "Deployment",
		},
	}

	newObj.SetName("b")
	newObj.SetNamespace("b")
	newObj.SetUID("b")
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
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

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

	// patch has 'apiVersion: apps/v1beta1' but live version is apps/v1 -> error
	err = f.Apply(appliedObj, "fieldmanager_test", false)
	if err == nil {
		t.Fatalf("expected an error from mismatched patch and live versions")
	}
	switch typ := err.(type) {
	default:
		t.Fatalf("expected error to be of type %T was %T (%v)", apierrors.StatusError{}, typ, err)
	case apierrors.APIStatus:
		if typ.Status().Code != http.StatusBadRequest {
			t.Fatalf("expected status code to be %d but was %d",
				http.StatusBadRequest, typ.Status().Code)
		}
	}
}

func TestVersionCheckDoesNotPanic(t *testing.T) {
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

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
		t.Fatalf("expected error to be of type %T was %T (%v)", apierrors.StatusError{}, typ, err)
	case apierrors.APIStatus:
		if typ.Status().Code != http.StatusBadRequest {
			t.Fatalf("expected status code to be %d but was %d",
				http.StatusBadRequest, typ.Status().Code)
		}
	}
}

func TestApplyDoesNotStripLabels(t *testing.T) {
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("v1", "Pod"))

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
	s, err := os.ReadFile(file)
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
			f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, test.gvk)

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

func TestApplyFailsWithManagedFields(t *testing.T) {
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("v1", "Pod"))

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
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("v1", "Pod"))

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
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("v1", "Pod"))

	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a": "b"
			},
			"creationTimestamp": null,
		}
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	if err := f.Apply(obj.DeepCopyObject(), "fieldmanager_test_apply", false); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}
	before := f.Live()
	// Wait to make sure the timestamp is different
	time.Sleep(time.Second)
	// Applying with a different fieldmanager will create an entry..
	if err := f.Apply(obj.DeepCopyObject(), "fieldmanager_test_apply_other", false); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	if reflect.DeepEqual(before, f.Live()) {
		t.Fatalf("Applying no-op apply with new manager didn't change object: \n%v", f.Live())
	}
	before = f.Live()
	// Wait to make sure the timestamp is different
	time.Sleep(time.Second)
	if err := f.Update(obj.DeepCopyObject(), "fieldmanager_test_update"); err != nil {
		t.Fatalf("failed to update object: %v", err)
	}
	if !reflect.DeepEqual(before, f.Live()) {
		t.Fatalf("No-op update has changed the object:\n%v\n---\n%v", before, f.Live())
	}
	before = f.Live()
	// Wait to make sure the timestamp is different
	time.Sleep(time.Second)
	if err := f.Apply(obj.DeepCopyObject(), "fieldmanager_test_apply", true); err != nil {
		t.Fatalf("failed to re-apply object: %v", err)
	}
	if !reflect.DeepEqual(before, f.Live()) {
		t.Fatalf("No-op apply has changed the object:\n%v\n---\n%v", before, f.Live())
	}
}

// Tests that one can reset the managedFields by sending either an empty
// list
func TestResetManagedFieldsEmptyList(t *testing.T) {
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("v1", "Pod"))

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
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("v1", "Pod"))

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
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

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
	if err := internal.SetLastApplied(newObj, invalidLastApplied); err != nil {
		t.Errorf("failed to set last applied: %v", err)
	}

	if err := f.Update(newObj, "kubectl-client-side-apply-test"); err != nil {
		t.Errorf("failed to update object: %v", err)
	}

	lastApplied, err := getLastApplied(f.Live())
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

	lastApplied, err = getLastApplied(f.Live())
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

	lastApplied, err = getLastApplied(f.Live())
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if lastApplied == invalidLastApplied ||
		!strings.Contains(lastApplied, "my-app-v2") {
		t.Errorf("expected last applied annotation to be updated, but got: %q", lastApplied)
	}
}

func TestInteropForClientSideApplyAndServerSideApply(t *testing.T) {
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

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
	lastApplied, err := getLastApplied(f.Live())
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

	lastApplied, err = getLastApplied(f.Live())
	if err != nil {
		t.Errorf("failed to get last applied: %v", err)
	}
	if !strings.Contains(lastApplied, "my-image-v2") {
		t.Errorf("expected last applied annotation to be updated, but got: %q", lastApplied)
	}
}

func TestNoTrackManagedFieldsForClientSideApply(t *testing.T) {
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter, schema.FromAPIVersionAndKind("apps/v1", "Deployment"))

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
	lastApplied, err := getLastApplied(f.Live())
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
	lastApplied, err = getLastApplied(f.Live())
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
	return internal.SetLastApplied(obj, lastAppliedJSON)
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

	lastApplied, ok := annotations[internal.LastAppliedConfigAnnotation]
	if !ok {
		return "", fmt.Errorf("expected last applied annotation, but got none for object: %v", obj)
	}
	return lastApplied, nil
}

func TestUpdateViaSubresources(t *testing.T) {
	f := managedfieldstest.NewTestFieldManagerSubresource(fakeTypeConverter, schema.FromAPIVersionAndKind("v1", "Pod"), "scale")

	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "v1",
		"kind": "Pod",
		"metadata": {
			"labels": {
				"a":"b"
			},
		}
	}`), &obj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}
	obj.SetManagedFields([]metav1.ManagedFieldsEntry{
		{
			Manager:    "test",
			Operation:  metav1.ManagedFieldsOperationApply,
			APIVersion: "apps/v1",
			FieldsType: "FieldsV1",
			FieldsV1: &metav1.FieldsV1{
				Raw: []byte(`{"f:metadata":{"f:labels":{"f:another_field":{}}}}`),
			},
		},
	})

	// Check that managed fields cannot be changed explicitly via subresources
	expectedManager := "fieldmanager_test_subresource"
	if err := f.Update(obj, expectedManager); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	managedFields := f.ManagedFields()
	if len(managedFields) != 1 {
		t.Fatalf("Expected new managed fields to have one entry. Got:\n%#v", managedFields)
	}
	if managedFields[0].Manager != expectedManager {
		t.Fatalf("Expected first item to have manager set to: %s. Got: %s", expectedManager, managedFields[0].Manager)
	}

	// Check that managed fields cannot be reset via subresources
	newObj := obj.DeepCopy()
	newObj.SetManagedFields([]metav1.ManagedFieldsEntry{})
	if err := f.Update(newObj, expectedManager); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}
	newManagedFields := f.ManagedFields()
	if len(newManagedFields) != 1 {
		t.Fatalf("Expected new managed fields to have one entry. Got:\n%#v", newManagedFields)
	}
}

// Ensures that a no-op Apply does not mutate managed fields
func TestApplyDoesNotChangeManagedFields(t *testing.T) {
	originalManagedFields := []metav1.ManagedFieldsEntry{}
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter,
		schema.FromAPIVersionAndKind("apps/v1", "Deployment"))
	newObj := &unstructured.Unstructured{
		Object: map[string]interface{}{},
	}
	appliedObj := &unstructured.Unstructured{
		Object: map[string]interface{}{},
	}

	// Convert YAML string inputs to unstructured instances
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
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
	}`), &newObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

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

	// Agent A applies initial configuration
	if err := f.Apply(newObj.DeepCopyObject(), "fieldmanager_z", false); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	// Agent B applies additive configuration
	if err := f.Apply(appliedObj, "fieldmanager_b", false); err != nil {
		t.Fatalf("failed to apply object %v", err)
	}

	// Next, agent A applies the initial configuration again, but we expect
	// a no-op to managed fields.
	//
	// The following update is expected not to change the liveObj, save off
	//	the fields
	for _, field := range f.ManagedFields() {
		originalManagedFields = append(originalManagedFields, *field.DeepCopy())
	}

	// Make sure timestamp change would be caught
	time.Sleep(2 * time.Second)

	if err := f.Apply(newObj, "fieldmanager_z", false); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	// ensure that the live object is unchanged
	if !reflect.DeepEqual(originalManagedFields, f.ManagedFields()) {
		originalYAML, _ := yaml.Marshal(originalManagedFields)
		current, _ := yaml.Marshal(f.ManagedFields())

		// should have been a no-op w.r.t. managed fields
		t.Fatalf("managed fields changed: ORIGINAL\n%v\nCURRENT\n%v",
			string(originalYAML), string(current))
	}
}

// Ensures that a no-op Update does not mutate managed fields
func TestUpdateDoesNotChangeManagedFields(t *testing.T) {
	originalManagedFields := []metav1.ManagedFieldsEntry{}
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter,
		schema.FromAPIVersionAndKind("apps/v1", "Deployment"))
	newObj := &unstructured.Unstructured{
		Object: map[string]interface{}{},
	}

	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
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
	}`), &newObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

	// Agent A updates with initial configuration
	if err := f.Update(newObj.DeepCopyObject(), "fieldmanager_z"); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	for _, field := range f.ManagedFields() {
		originalManagedFields = append(originalManagedFields, *field.DeepCopy())
	}

	// Make sure timestamp change would be caught
	time.Sleep(2 * time.Second)

	// If the same exact configuration is updated once again, the
	// managed fields are not expected to change
	//
	// However, a change in field ownership WOULD be a semantic change which
	// should cause managed fields to change.
	if err := f.Update(newObj, "fieldmanager_z"); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	// ensure that the live object is unchanged
	if !reflect.DeepEqual(originalManagedFields, f.ManagedFields()) {
		originalYAML, _ := yaml.Marshal(originalManagedFields)
		current, _ := yaml.Marshal(f.ManagedFields())

		// should have been a no-op w.r.t. managed fields
		t.Fatalf("managed fields changed: ORIGINAL\n%v\nCURRENT\n%v",
			string(originalYAML), string(current))
	}
}

// This test makes sure that the liveObject during a patch does not mutate
// its managed fields.
func TestLiveObjectManagedFieldsNotRemoved(t *testing.T) {
	f := managedfieldstest.NewTestFieldManager(fakeTypeConverter,
		schema.FromAPIVersionAndKind("apps/v1", "Deployment"))
	newObj := &unstructured.Unstructured{
		Object: map[string]interface{}{},
	}
	appliedObj := &unstructured.Unstructured{
		Object: map[string]interface{}{},
	}
	// Convert YAML string inputs to unstructured instances
	if err := yaml.Unmarshal([]byte(`{
		"apiVersion": "apps/v1",
		"kind": "Deployment",
		"metadata": {
			"name": "deployment",
			"labels": {"app": "nginx"}
		},
		"spec": {
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
	}`), &newObj.Object); err != nil {
		t.Fatalf("error decoding YAML: %v", err)
	}

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

	// Agent A applies initial configuration
	if err := f.Apply(newObj.DeepCopyObject(), "fieldmanager_z", false); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	originalLiveObj := f.Live()

	accessor, err := meta.Accessor(originalLiveObj)
	if err != nil {
		panic(fmt.Errorf("couldn't get accessor: %v", err))
	}

	// Managed fields should not be stripped
	if len(accessor.GetManagedFields()) == 0 {
		t.Fatalf("empty managed fields of object which expected nonzero fields")
	}

	// Agent A applies the exact same configuration
	if err := f.Apply(appliedObj.DeepCopyObject(), "fieldmanager_z", false); err != nil {
		t.Fatalf("failed to apply object: %v", err)
	}

	accessor, err = meta.Accessor(originalLiveObj)
	if err != nil {
		panic(fmt.Errorf("couldn't get accessor: %v", err))
	}

	// Managed fields should not be stripped
	if len(accessor.GetManagedFields()) == 0 {
		t.Fatalf("empty managed fields of object which expected nonzero fields")
	}
}
