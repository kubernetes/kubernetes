/*
Copyright 2017 The Kubernetes Authors.

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

package node

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
)

func TestValidateAPIServer(t *testing.T) {
	expect := version.Info{
		Major:     "foo",
		Minor:     "bar",
		GitCommit: "baz",
	}
	tests := []struct {
		s      *httptest.Server
		expect bool
	}{
		{
			s:      httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {})),
			expect: false,
		},
		{
			s: httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				var obj interface{}
				switch req.URL.Path {
				case "/api":
					obj = &metav1.APIVersions{
						Versions: []string{
							"v1.6.0",
						},
					}
					output, err := json.Marshal(obj)
					if err != nil {
						t.Fatalf("unexpected encoding error: %v", err)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					w.Write(output)
				default:
					output, err := json.Marshal(expect)
					if err != nil {
						t.Errorf("unexpected encoding error: %v", err)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					w.Write(output)
				}
			})),
			expect: false,
		},
		{
			s: httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
				var obj interface{}
				switch req.URL.Path {
				case "/api":
					obj = &metav1.APIVersions{
						Versions: []string{
							"v1.6.0",
						},
					}
					output, err := json.Marshal(obj)
					if err != nil {
						t.Fatalf("unexpected encoding error: %v", err)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					w.Write(output)
				case "/apis":
					obj = &metav1.APIGroupList{
						Groups: []metav1.APIGroup{
							{
								Name: "certificates.k8s.io",
								Versions: []metav1.GroupVersionForDiscovery{
									{GroupVersion: "certificates.k8s.io/v1beta1", Version: "v1beta1"},
								},
							},
						},
					}
					output, err := json.Marshal(obj)
					if err != nil {
						t.Fatalf("unexpected encoding error: %v", err)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					w.Write(output)
				default:
					output, err := json.Marshal(expect)
					if err != nil {
						t.Errorf("unexpected encoding error: %v", err)
						return
					}
					w.Header().Set("Content-Type", "application/json")
					w.WriteHeader(http.StatusOK)
					w.Write(output)
				}
			})),
			expect: true,
		},
	}
	for _, rt := range tests {
		defer rt.s.Close()
		rc := &restclient.Config{Host: rt.s.URL}
		c, err := discovery.NewDiscoveryClientForConfig(rc)
		if err != nil {
			t.Fatalf("encountered an error while trying to get the new discovery client: %v", err)
		}
		cs := &clientset.Clientset{DiscoveryClient: c}
		actual := ValidateAPIServer(cs)
		if (actual == nil) != rt.expect {
			t.Errorf(
				"failed TestValidateAPIServer:\n\texpected: %t\n\t  actual: %t",
				rt.expect,
				(actual == nil),
			)
		}
	}
}
