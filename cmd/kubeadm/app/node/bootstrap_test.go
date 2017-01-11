/*
Copyright 2016 The Kubernetes Authors.

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
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/client/typed/discovery"
	"k8s.io/kubernetes/pkg/version"
)

func TestEstablishMasterConnection(t *testing.T) {
	expect := version.Info{
		Major:     "foo",
		Minor:     "bar",
		GitCommit: "baz",
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		var obj interface{}
		switch req.URL.Path {
		case "/api":
			obj = &metav1.APIVersions{
				Versions: []string{
					"v1.4",
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
							{GroupVersion: "extensions/v1beta1"},
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
	}))
	defer srv.Close()

	tests := []struct {
		c      string
		e      string
		expect bool
	}{
		{
			c:      "",
			e:      "",
			expect: false,
		},
		{
			c:      "",
			e:      srv.URL,
			expect: true,
		},
		{
			c:      "foo",
			e:      srv.URL,
			expect: true,
		},
	}
	for _, rt := range tests {
		s := &kubeadmapi.TokenDiscovery{}
		c := &kubeadmapi.ClusterInfo{Endpoints: []string{rt.e}, CertificateAuthorities: []string{rt.c}}
		_, actual := EstablishMasterConnection(s, c)
		if (actual == nil) != rt.expect {
			t.Errorf(
				"failed EstablishMasterConnection:\n\texpected: %t\n\t  actual: %t",
				rt.expect,
				(actual == nil),
			)
		}
	}
}

func TestCreateClients(t *testing.T) {
	tests := []struct {
		e      string
		expect bool
	}{
		{
			e:      "",
			expect: false,
		},
		{
			e:      "foo",
			expect: true,
		},
	}
	for _, rt := range tests {
		_, actual := createClients(nil, rt.e, "", "")
		if (actual == nil) != rt.expect {
			t.Errorf(
				"failed createClients:\n\texpected: %t\n\t  actual: %t",
				rt.expect,
				(actual == nil),
			)
		}
	}
}

func TestCheckAPIEndpoint(t *testing.T) {
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
							"v1.4",
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
							"v1.4",
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
									{GroupVersion: "extensions/v1beta1"},
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
			t.Fatalf("encountered an error while trying to get New Discovery Client: %v", err)
		}
		cs := &clientset.Clientset{DiscoveryClient: c}
		actual := checkAPIEndpoint(cs, "")
		if (actual == nil) != rt.expect {
			t.Errorf(
				"failed runChecks:\n\texpected: %t\n\t  actual: %t",
				rt.expect,
				(actual == nil),
			)
		}
	}
}
