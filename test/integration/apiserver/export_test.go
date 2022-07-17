/*
Copyright 2020 The Kubernetes Authors.

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
	"net/http"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Tests that the apiserver rejects the export param
func TestExportRejection(t *testing.T) {
	clientSet, _, tearDownFn := setup(t)
	defer tearDownFn()

	_, err := clientSet.CoreV1().Namespaces().Create(context.Background(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{Name: "export-fail"},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		clientSet.CoreV1().Namespaces().Delete(context.Background(), "export-fail", metav1.DeleteOptions{})
	}()

	result := clientSet.Discovery().RESTClient().Get().AbsPath("/api/v1/namespaces/export-fail").Param("export", "true").Do(context.Background())
	statusCode := 0
	result.StatusCode(&statusCode)
	if statusCode != http.StatusBadRequest {
		t.Errorf("expected %v, got %v", http.StatusBadRequest, statusCode)
	}

	result = clientSet.Discovery().RESTClient().Get().AbsPath("/api/v1/namespaces/export-fail").Param("export", "false").Do(context.Background())
	statusCode = 0
	result.StatusCode(&statusCode)
	if statusCode != http.StatusOK {
		t.Errorf("expected %v, got %v", http.StatusOK, statusCode)
	}

}
