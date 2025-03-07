/*
Copyright 2024 The Kubernetes Authors.

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

package tests

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/google/go-cmp/cmp"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	corev1 "k8s.io/api/core/v1"
	crv1 "k8s.io/apiextensions-apiserver/examples/client-go/pkg/apis/cr/v1"
	crv1client "k8s.io/apiextensions-apiserver/examples/client-go/pkg/client/clientset/versioned"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

func TestClientContentType(t *testing.T) {
	createPodFunc := func(t *testing.T, config *rest.Config) {
		client, err := kubernetes.NewForConfig(config)
		if err != nil {
			t.Fatalf("failed to create REST client: %v", err)
		}

		_, err = client.CoreV1().Pods("panda").
			Create(context.TODO(), &corev1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "snorlax"}}, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}

	updateScaleFunc := func(t *testing.T, config *rest.Config) {
		client, err := kubernetes.NewForConfig(config)
		if err != nil {
			t.Fatalf("failed to create REST client: %v", err)
		}

		_, err = client.AppsV1().Deployments("panda").
			UpdateScale(context.TODO(), "snorlax", &autoscalingv1.Scale{ObjectMeta: metav1.ObjectMeta{Name: "snorlax"}}, metav1.UpdateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}

	createExampleViaRESTClientFunc := func(t *testing.T, config *rest.Config) {
		kubeClient, err := kubernetes.NewForConfig(config)
		if err != nil {
			t.Fatalf("failed to create REST client: %v", err)
		}

		client := crv1client.New(kubeClient.CoreV1().RESTClient())

		_, err = client.CrV1().Examples("panda").
			Create(context.TODO(), &crv1.Example{ObjectMeta: metav1.ObjectMeta{Name: "snorlax"}}, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}

	tests := []struct {
		name              string
		createFunc        func(*testing.T, *rest.Config)
		contentType       string
		expectedPath      string
		expectContentType string
		expectBody        string
	}{
		{
			name:              "default",
			createFunc:        createPodFunc,
			contentType:       "",
			expectedPath:      "/api/v1/namespaces/panda/pods",
			expectContentType: "application/vnd.kubernetes.protobuf",
			expectBody:        "k8s\x00\n\t\n\x02v1\x12\x03Pod\x12\x4c\n\x17\n\asnorlax\x12\x00\x1a\x00\"\x00*\x002\x008\x00B\x00\x12\x1c\x1a\x002\x00B\x00J\x00R\x00X\x00`\x00h\x00\x82\x01\x00\x8a\x01\x00\x9a\x01\x00\xc2\x01\x00\x1a\x13\n\x00\x1a\x00\"\x00*\x002\x00J\x00Z\x00r\x00\x88\x01\x00\x1a\x00\"\x00",
		},
		{
			name:              "json",
			createFunc:        createPodFunc,
			contentType:       "application/json",
			expectedPath:      "/api/v1/namespaces/panda/pods",
			expectContentType: "application/json",
			expectBody: `{"kind":"Pod","apiVersion":"v1","metadata":{"name":"snorlax","creationTimestamp":null},"spec":{"containers":null},"status":{}}
`,
		},
		{
			name:              "default update scale",
			createFunc:        updateScaleFunc,
			contentType:       "",
			expectedPath:      "/apis/apps/v1/namespaces/panda/deployments/snorlax/scale",
			expectContentType: "application/vnd.kubernetes.protobuf",
			expectBody:        "k8s\u0000\n\u0017\n\u000Eautoscaling/v1\u0012\u0005Scale\u0012#\n\u0017\n\asnorlax\u0012\u0000\u001A\u0000\"\u0000*\u00002\u00008\u0000B\u0000\u0012\u0002\b\u0000\u001A\u0004\b\u0000\u0012\u0000\u001A\u0000\"\u0000",
		},
		{
			name:              "json update scale",
			createFunc:        updateScaleFunc,
			contentType:       "application/json",
			expectedPath:      "/apis/apps/v1/namespaces/panda/deployments/snorlax/scale",
			expectContentType: "application/json",
			expectBody: `{"kind":"Scale","apiVersion":"autoscaling/v1","metadata":{"name":"snorlax","creationTimestamp":null},"spec":{},"status":{"replicas":0}}
`,
		},
		{
			name:              "default via RESTClient",
			createFunc:        createExampleViaRESTClientFunc,
			contentType:       "",
			expectedPath:      "/api/v1/namespaces/panda/examples",
			expectContentType: "application/json",
			expectBody: `{"metadata":{"name":"snorlax","creationTimestamp":null},"spec":{"foo":"","bar":false},"status":{}}
`,
		},
		{
			name:              "json via RESTClient",
			createFunc:        createExampleViaRESTClientFunc,
			contentType:       "application/json",
			expectedPath:      "/api/v1/namespaces/panda/examples",
			expectContentType: "application/json",
			expectBody: `{"metadata":{"name":"snorlax","creationTimestamp":null},"spec":{"foo":"","bar":false},"status":{}}
`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var calls atomic.Uint64
			ts := httptest.NewUnstartedServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				calls.Add(1)

				if got, want := r.URL.Path, tc.expectedPath; got != want {
					t.Errorf("unexpected path, got=%q, want=%q", got, want)
				}

				if got, want := r.Header.Get("content-type"), tc.expectContentType; got != want {
					t.Errorf("unexpected content-type, got=%q, want=%q", got, want)
				}

				if r.Body == nil {
					t.Fatal("request body is nil")
				}
				body, err := io.ReadAll(r.Body)
				if err != nil {
					t.Fatal(err)
				}
				_ = r.Body.Close()
				if diff := cmp.Diff(tc.expectBody, string(body)); len(diff) > 0 {
					t.Errorf("body diff (-want, +got):\n%s", diff)
				}

				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte("{}"))
			}))
			ts.Start()
			defer ts.Close()

			config := &rest.Config{
				Host:          ts.URL,
				ContentConfig: rest.ContentConfig{ContentType: tc.contentType},
			}

			tc.createFunc(t, config)

			if calls.Load() != 1 {
				t.Errorf("unexpected handler call count: %d", calls.Load())
			}
		})
	}
}
