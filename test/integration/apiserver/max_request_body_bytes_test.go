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

package apiserver

import (
	"context"
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/test/integration/framework"
)

// Tests that the apiserver limits the resource size in write operations.
func TestMaxResourceSize(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)
	clientSet, _ := framework.StartTestServer(t, stopCh, framework.TestServerSetup{})

	hugeData := []byte(strings.Repeat("x", 3*1024*1024+1))

	rest := clientSet.Discovery().RESTClient()

	c := clientSet.CoreV1().RESTClient()
	t.Run("Create should limit the request body size", func(t *testing.T) {
		err := c.Post().AbsPath(fmt.Sprintf("/api/v1/namespaces/default/pods")).
			Body(hugeData).Do(context.TODO()).Error()
		if err == nil {
			t.Fatalf("unexpected no error")
		}
		if !apierrors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected requested entity too large err, got %v", err)

		}
	})

	// Create a secret so we can update/patch/delete it.
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
	}
	_, err := clientSet.CoreV1().Secrets("default").Create(context.TODO(), secret, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	t.Run("Update should limit the request body size", func(t *testing.T) {
		err = c.Put().AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(hugeData).Do(context.TODO()).Error()
		if err == nil {
			t.Fatalf("unexpected no error")
		}
		if !apierrors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected requested entity too large err, got %v", err)

		}
	})
	t.Run("Patch should limit the request body size", func(t *testing.T) {
		err = c.Patch(types.JSONPatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(hugeData).Do(context.TODO()).Error()
		if err == nil {
			t.Fatalf("unexpected no error")
		}
		if !apierrors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected requested entity too large err, got %v", err)

		}
	})
	t.Run("JSONPatchType should handle a patch just under the max limit", func(t *testing.T) {
		patchBody := []byte(`[{"op":"add","path":"/foo","value":` + strings.Repeat("[", 3*1024*1024/2-100) + strings.Repeat("]", 3*1024*1024/2-100) + `}]`)
		err = rest.Patch(types.JSONPatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(patchBody).Do(context.TODO()).Error()
		if err != nil && !apierrors.IsBadRequest(err) {
			t.Errorf("expected success or bad request err, got %v", err)
		}
	})
	t.Run("JSONPatchType should handle a valid patch just under the max limit", func(t *testing.T) {
		patchBody := []byte(`[{"op":"add","path":"/foo","value":0` + strings.Repeat(" ", 3*1024*1024-100) + `}]`)
		err = rest.Patch(types.JSONPatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(patchBody).Do(context.TODO()).Error()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
	t.Run("MergePatchType should handle a patch just under the max limit", func(t *testing.T) {
		patchBody := []byte(`{"value":` + strings.Repeat("[", 3*1024*1024/2-100) + strings.Repeat("]", 3*1024*1024/2-100) + `}`)
		err = rest.Patch(types.MergePatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(patchBody).Do(context.TODO()).Error()
		if err != nil && !apierrors.IsBadRequest(err) {
			t.Errorf("expected success or bad request err, got %v", err)
		}
	})
	t.Run("MergePatchType should handle a valid patch just under the max limit", func(t *testing.T) {
		patchBody := []byte(`{"value":0` + strings.Repeat(" ", 3*1024*1024-100) + `}`)
		err = rest.Patch(types.MergePatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(patchBody).Do(context.TODO()).Error()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
	t.Run("StrategicMergePatchType should handle a patch just under the max limit", func(t *testing.T) {
		patchBody := []byte(`{"value":` + strings.Repeat("[", 3*1024*1024/2-100) + strings.Repeat("]", 3*1024*1024/2-100) + `}`)
		err = rest.Patch(types.StrategicMergePatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(patchBody).Do(context.TODO()).Error()
		if err != nil && !apierrors.IsBadRequest(err) {
			t.Errorf("expected success or bad request err, got %v", err)
		}
	})
	t.Run("StrategicMergePatchType should handle a valid patch just under the max limit", func(t *testing.T) {
		patchBody := []byte(`{"value":0` + strings.Repeat(" ", 3*1024*1024-100) + `}`)
		err = rest.Patch(types.StrategicMergePatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(patchBody).Do(context.TODO()).Error()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
	t.Run("ApplyPatchType should handle a patch just under the max limit", func(t *testing.T) {
		patchBody := []byte(`{"value":` + strings.Repeat("[", 3*1024*1024/2-100) + strings.Repeat("]", 3*1024*1024/2-100) + `}`)
		err = rest.Patch(types.ApplyPatchType).Param("fieldManager", "test").AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(patchBody).Do(context.TODO()).Error()
		if err != nil && !apierrors.IsBadRequest(err) {
			t.Errorf("expected success or bad request err, got %#v", err)
		}
	})
	t.Run("ApplyPatchType should handle a valid patch just under the max limit", func(t *testing.T) {
		patchBody := []byte(`{"apiVersion":"v1","kind":"Secret"` + strings.Repeat(" ", 3*1024*1024-100) + `}`)
		err = rest.Patch(types.ApplyPatchType).Param("fieldManager", "test").AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(patchBody).Do(context.TODO()).Error()
		if err != nil {
			t.Errorf("unexpected error: %v", err)
		}
	})
	t.Run("Delete should limit the request body size", func(t *testing.T) {
		err = c.Delete().AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(hugeData).Do(context.TODO()).Error()
		if err == nil {
			t.Fatalf("unexpected no error")
		}
		if !apierrors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected requested entity too large err, got %v", err)

		}
	})

	// Create YAML over 3MB limit
	t.Run("create should limit yaml parsing", func(t *testing.T) {
		yamlBody := []byte(`
apiVersion: v1
kind: ConfigMap
metadata:
  name: mytest
values: ` + strings.Repeat("[", 3*1024*1024))

		_, err := rest.Post().
			SetHeader("Accept", "application/yaml").
			SetHeader("Content-Type", "application/yaml").
			AbsPath("/api/v1/namespaces/default/configmaps").
			Body(yamlBody).
			DoRaw(context.TODO())
		if !apierrors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected too large error, got %v", err)
		}
	})

	// Create YAML just under 3MB limit, nested
	t.Run("create should handle a yaml document just under the maximum size with correct nesting", func(t *testing.T) {
		yamlBody := []byte(`
apiVersion: v1
kind: ConfigMap
metadata:
  name: mytest
values: ` + strings.Repeat("[", 3*1024*1024/2-500) + strings.Repeat("]", 3*1024*1024/2-500))

		_, err := rest.Post().
			SetHeader("Accept", "application/yaml").
			SetHeader("Content-Type", "application/yaml").
			AbsPath("/api/v1/namespaces/default/configmaps").
			Body(yamlBody).
			DoRaw(context.TODO())
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request, got %v", err)
		}
	})

	// Create YAML just under 3MB limit, not nested
	t.Run("create should handle a yaml document just under the maximum size with unbalanced nesting", func(t *testing.T) {
		yamlBody := []byte(`
apiVersion: v1
kind: ConfigMap
metadata:
  name: mytest
values: ` + strings.Repeat("[", 3*1024*1024-1000))

		_, err := rest.Post().
			SetHeader("Accept", "application/yaml").
			SetHeader("Content-Type", "application/yaml").
			AbsPath("/api/v1/namespaces/default/configmaps").
			Body(yamlBody).
			DoRaw(context.TODO())
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request, got %v", err)
		}
	})

	// Create JSON over 3MB limit
	t.Run("create should limit json parsing", func(t *testing.T) {
		jsonBody := []byte(`{
	"apiVersion": "v1",
	"kind": "ConfigMap",
	"metadata": {
	  "name": "mytest"
	},
	"values": ` + strings.Repeat("[", 3*1024*1024/2) + strings.Repeat("]", 3*1024*1024/2) + "}")

		_, err := rest.Post().
			SetHeader("Accept", "application/json").
			SetHeader("Content-Type", "application/json").
			AbsPath("/api/v1/namespaces/default/configmaps").
			Body(jsonBody).
			DoRaw(context.TODO())
		if !apierrors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected too large error, got %v", err)
		}
	})

	// Create JSON just under 3MB limit, nested
	t.Run("create should handle a json document just under the maximum size with correct nesting", func(t *testing.T) {
		jsonBody := []byte(`{
	"apiVersion": "v1",
	"kind": "ConfigMap",
	"metadata": {
	  "name": "mytest"
	},
	"values": ` + strings.Repeat("[", 3*1024*1024/2-100) + strings.Repeat("]", 3*1024*1024/2-100) + "}")

		_, err := rest.Post().
			SetHeader("Accept", "application/json").
			SetHeader("Content-Type", "application/json").
			AbsPath("/api/v1/namespaces/default/configmaps").
			Body(jsonBody).
			DoRaw(context.TODO())
		// TODO(liggitt): expect bad request on deep nesting, rather than success on dropped unknown field data
		if err != nil && !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request, got %v", err)
		}
	})

	// Create JSON just under 3MB limit, not nested
	t.Run("create should handle a json document just under the maximum size with unbalanced nesting", func(t *testing.T) {
		jsonBody := []byte(`{
	"apiVersion": "v1",
	"kind": "ConfigMap",
	"metadata": {
	  "name": "mytest"
	},
	"values": ` + strings.Repeat("[", 3*1024*1024-1000) + "}")

		_, err := rest.Post().
			SetHeader("Accept", "application/json").
			SetHeader("Content-Type", "application/json").
			AbsPath("/api/v1/namespaces/default/configmaps").
			Body(jsonBody).
			DoRaw(context.TODO())
		if !apierrors.IsBadRequest(err) {
			t.Errorf("expected bad request, got %v", err)
		}
	})
}
