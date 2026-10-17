/*
Copyright The Kubernetes Authors.

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

package webhook

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	authorizationv1 "k8s.io/api/authorization/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/rest"
)

func TestSubjectAccessReviewResponseSizeLimit(t *testing.T) {
	const oversized = 11 * 1024 * 1024 // larger than the 10MiB response limit

	tests := []struct {
		name    string
		version string
	}{
		{name: "v1", version: "v1"},
		{name: "v1beta1", version: "v1beta1"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"status":{"allowed":true,"reason":"`))
				_, _ = w.Write([]byte(strings.Repeat("a", oversized)))
				_, _ = w.Write([]byte(`"}}`))
			}))
			defer server.Close()

			client, err := subjectAccessReviewInterfaceFromConfig(&rest.Config{Host: server.URL}, tc.version, wait.Backoff{Steps: 1, Duration: time.Millisecond})
			if err != nil {
				t.Fatal(err)
			}
			sar := &authorizationv1.SubjectAccessReview{
				Spec: authorizationv1.SubjectAccessReviewSpec{User: "alice"},
			}
			_, _, err = client.Create(context.Background(), sar, metav1.CreateOptions{})
			if err == nil {
				t.Fatalf("expected an error for a response larger than the limit, got none")
			}
		})
	}
}
