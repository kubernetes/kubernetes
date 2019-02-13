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
	"fmt"
	"strings"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

// Tests that the apiserver limits the resource size in write operations.
func TestMaxResourceSize(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)
	clientSet, _ := framework.StartTestServer(t, stopCh, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024
		},
	})

	hugeData := []byte(strings.Repeat("x", 1024*1024+1))

	c := clientSet.CoreV1().RESTClient()
	t.Run("Create should limit the request body size", func(t *testing.T) {
		err := c.Post().AbsPath(fmt.Sprintf("/api/v1/namespaces/default/pods")).
			Body(hugeData).Do().Error()
		if err == nil {
			t.Fatalf("unexpected no error")
		}
		if !errors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected requested entity too large err, got %v", err)

		}
	})

	// Create a secret so we can update/patch/delete it.
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
	}
	_, err := clientSet.CoreV1().Secrets("default").Create(secret)
	if err != nil {
		t.Fatal(err)
	}

	t.Run("Update should limit the request body size", func(t *testing.T) {
		err = c.Put().AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(hugeData).Do().Error()
		if err == nil {
			t.Fatalf("unexpected no error")
		}
		if !errors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected requested entity too large err, got %v", err)

		}
	})
	t.Run("Patch should limit the request body size", func(t *testing.T) {
		err = c.Patch(types.JSONPatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(hugeData).Do().Error()
		if err == nil {
			t.Fatalf("unexpected no error")
		}
		if !errors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected requested entity too large err, got %v", err)

		}
	})
	t.Run("Delete should limit the request body size", func(t *testing.T) {
		err = c.Delete().AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
			Body(hugeData).Do().Error()
		if err == nil {
			t.Fatalf("unexpected no error")
		}
		if !errors.IsRequestEntityTooLargeError(err) {
			t.Errorf("expected requested entity too large err, got %v", err)

		}
	})
}
