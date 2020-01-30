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
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/test/integration/framework"
)

// Tests that the apiserver limits the number of operations in a json patch.
func TestMaxJSONPatchOperations(t *testing.T) {
	stopCh := make(chan struct{})
	defer close(stopCh)
	clientSet, _ := framework.StartTestServer(t, stopCh, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.GenericServerRunOptions.MaxRequestBodyBytes = 1024 * 1024
		},
	})

	p := `{"op":"add","path":"/x","value":"y"}`
	// maxJSONPatchOperations = 10000
	hugePatch := []byte("[" + strings.Repeat(p+",", 10000) + p + "]")

	c := clientSet.CoreV1().RESTClient()
	// Create a secret so we can patch it.
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test",
		},
	}
	_, err := clientSet.CoreV1().Secrets("default").Create(secret)
	if err != nil {
		t.Fatal(err)
	}

	err = c.Patch(types.JSONPatchType).AbsPath(fmt.Sprintf("/api/v1/namespaces/default/secrets/test")).
		Body(hugePatch).Do(context.TODO()).Error()
	if err == nil {
		t.Fatalf("unexpected no error")
	}
	if !apierrors.IsRequestEntityTooLargeError(err) {
		t.Errorf("expected requested entity too large err, got %v", err)
	}
	if !strings.Contains(err.Error(), "The allowed maximum operations in a JSON patch is") {
		t.Errorf("expected the error message to be about maximum operations, got %v", err)
	}
}
