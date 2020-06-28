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

package patchnode

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestAnnotateCRISocket(t *testing.T) {
	tests := []struct {
		name                       string
		currentCRISocketAnnotation string
		newCRISocketAnnotation     string
		expectedPatch              string
	}{
		{
			name:                       "CRI-socket annotation missing",
			currentCRISocketAnnotation: "",
			newCRISocketAnnotation:     "/run/containerd/containerd.sock",
			expectedPatch:              `{"metadata":{"annotations":{"kubeadm.alpha.kubernetes.io/cri-socket":"/run/containerd/containerd.sock"}}}`,
		},
		{
			name:                       "CRI-socket annotation already exists",
			currentCRISocketAnnotation: "/run/containerd/containerd.sock",
			newCRISocketAnnotation:     "/run/containerd/containerd.sock",
			expectedPatch:              `{}`,
		},
		{
			name:                       "CRI-socket annotation needs to be updated",
			currentCRISocketAnnotation: "/var/run/dockershim.sock",
			newCRISocketAnnotation:     "/run/containerd/containerd.sock",
			expectedPatch:              `{"metadata":{"annotations":{"kubeadm.alpha.kubernetes.io/cri-socket":"/run/containerd/containerd.sock"}}}`,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			nodename := "node01"
			node := &v1.Node{
				ObjectMeta: metav1.ObjectMeta{
					Name: nodename,
					Labels: map[string]string{
						v1.LabelHostname: nodename,
					},
					Annotations: map[string]string{},
				},
			}

			if tc.currentCRISocketAnnotation != "" {
				node.ObjectMeta.Annotations[kubeadmconstants.AnnotationKubeadmCRISocket] = tc.currentCRISocketAnnotation
			}

			jsonNode, err := json.Marshal(node)
			if err != nil {
				t.Fatalf("unexpected encoding error: %v", err)
			}

			var patchRequest string
			s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				w.Header().Set("Content-Type", "application/json")

				if req.URL.Path != "/api/v1/nodes/"+nodename {
					t.Errorf("request for unexpected HTTP resource: %v", req.URL.Path)
					http.Error(w, "", http.StatusNotFound)
					return
				}

				switch req.Method {
				case "GET":
				case "PATCH":
					buf := new(bytes.Buffer)
					buf.ReadFrom(req.Body)
					patchRequest = buf.String()
				default:
					t.Errorf("request for unexpected HTTP verb: %v", req.Method)
					http.Error(w, "", http.StatusNotFound)
					return
				}

				w.WriteHeader(http.StatusOK)
				w.Write(jsonNode)
			}))
			defer s.Close()

			cs, err := clientset.NewForConfig(&restclient.Config{Host: s.URL})
			if err != nil {
				t.Fatalf("unexpected error building clientset: %v", err)
			}

			if err := AnnotateCRISocket(cs, nodename, tc.newCRISocketAnnotation); err != nil {
				t.Errorf("unexpected error: %v", err)
			}

			if tc.expectedPatch != patchRequest {
				t.Errorf("expected patch %v, got %v", tc.expectedPatch, patchRequest)
			}
		})
	}
}
