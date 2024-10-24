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

package apiserver

import (
	"context"
	"net/http"
	"sync/atomic"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

type roundTripperFunc func(request *http.Request) (*http.Response, error)

func (f roundTripperFunc) RoundTrip(request *http.Request) (*http.Response, error) {
	return f(request)
}

func TestCBORNegotiatedResponse(t *testing.T) {
	DoGeneratedClient := func(accept string) func(t *testing.T, config *rest.Config) string {
		return func(t *testing.T, config *rest.Config) string {
			var (
				roundtrips uint64
				got        string
			)

			config.AcceptContentTypes = accept
			config.Wrap(func(rt http.RoundTripper) http.RoundTripper {
				return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
					atomic.AddUint64(&roundtrips, 1)
					response, err := rt.RoundTrip(request)
					if response != nil {
						got = response.Header.Get("Content-Type")
					}
					return response, err
				})
			})
			client, err := kubernetes.NewForConfig(config)
			if err != nil {
				t.Fatal(err)
			}

			if _, err := client.CoreV1().Namespaces().List(context.TODO(), metav1.ListOptions{}); err != nil {
				t.Fatal(err)
			}

			if n := atomic.LoadUint64(&roundtrips); n != 1 {
				t.Fatalf("expected 1 roundtrip, saw %d", n)
			}

			return got
		}
	}

	DoDynamicClient := func(allowed bool) func(t *testing.T, config *rest.Config) string {
		return func(t *testing.T, config *rest.Config) string {
			var (
				roundtrips uint64
				got        string
			)

			config.ContentType = "application/json" // this test only cares about response negotiation
			config.Wrap(func(rt http.RoundTripper) http.RoundTripper {
				return roundTripperFunc(func(request *http.Request) (*http.Response, error) {
					atomic.AddUint64(&roundtrips, 1)
					response, err := rt.RoundTrip(request)
					if response != nil {
						got = response.Header.Get("Content-Type")
					}
					return response, err
				})
			})

			var (
				client dynamic.Interface
				err    error
			)
			func() {
				if !allowed {
					// TODO: Client feature gate will control disabling CBOR.
					original := dynamic.AcceptContentTypes
					dynamic.AcceptContentTypes = "application/json"
					defer func() { dynamic.AcceptContentTypes = original }()
				}

				client, err = dynamic.NewForConfig(config)
				if err != nil {
					t.Fatal(err)
				}
			}()

			if client.Resource(corev1.SchemeGroupVersion.WithResource("namespaces")).List(context.TODO(), metav1.ListOptions{}); err != nil {
				t.Fatal(err)
			}

			if n := atomic.LoadUint64(&roundtrips); n != 1 {
				t.Fatalf("expected 1 roundtrip, saw %d", n)
			}

			return got
		}
	}

	for _, tc := range []struct {
		Name              string
		EnableCBOR        bool
		ExpectContentType string
		DoRequest         func(t *testing.T, config *rest.Config) string
	}{
		{
			Name:              "generated client cbor disabled protobuf first",
			EnableCBOR:        false,
			ExpectContentType: "application/vnd.kubernetes.protobuf",
			DoRequest:         DoGeneratedClient("application/vnd.kubernetes.protobuf;q=1,application/cbor;q=0.9,application/json;q=0.8"),
		},
		{
			Name:              "generated client cbor disabled cbor first",
			EnableCBOR:        false,
			ExpectContentType: "application/json",
			DoRequest:         DoGeneratedClient("application/cbor;q=1,application/json;q=0.9"),
		},
		{
			Name:              "generated client cbor disabled cbor preferred",
			EnableCBOR:        false,
			ExpectContentType: "application/json",
			DoRequest:         DoGeneratedClient("application/cbor;q=1,application/json;q=0.9"),
		},
		{
			Name:              "generated client cbor enabled protobuf preferred",
			EnableCBOR:        true,
			ExpectContentType: "application/vnd.kubernetes.protobuf",
			DoRequest:         DoGeneratedClient("application/vnd.kubernetes.protobuf;q=1,application/cbor;q=0.9,application/json;q=0.8"),
		},
		{
			Name:              "generated client cbor enabled cbor preferred",
			EnableCBOR:        true,
			ExpectContentType: "application/cbor",
			DoRequest:         DoGeneratedClient("application/cbor;q=1,application/json;q=0.9"),
		},
		{
			Name:              "generated client cbor enabled json preferred",
			EnableCBOR:        true,
			ExpectContentType: "application/json",
			DoRequest:         DoGeneratedClient("application/cbor;q=0.9,application/json;q=1"),
		},
		{
			Name:              "dynamic client cbor disabled",
			EnableCBOR:        false,
			ExpectContentType: "application/json",
			DoRequest:         DoDynamicClient(true),
		},
		{
			Name:              "dynamic client cbor enabled cbor preferred",
			EnableCBOR:        true,
			ExpectContentType: "application/cbor",
			DoRequest:         DoDynamicClient(true),
		},
		{
			Name:              "dynamic client cbor enabled json preferred",
			EnableCBOR:        true,
			ExpectContentType: "application/json",
			DoRequest:         DoDynamicClient(false),
		},
	} {
		t.Run(tc.Name, func(t *testing.T) {
			if tc.EnableCBOR {
				framework.EnableCBORForTest(t)
			}

			server := kubeapiservertesting.StartTestServerOrDie(t, nil, nil, framework.SharedEtcd())
			defer server.TearDownFn()

			got := tc.DoRequest(t, rest.CopyConfig(server.ClientConfig))

			if got != tc.ExpectContentType {
				t.Errorf("expected response content type %q, got %q", tc.ExpectContentType, got)
			}
		})
	}
}
