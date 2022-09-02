/*
Copyright 2022 The Kubernetes Authors.

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
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/emicklei/go-restful/v3"
	fuzz "github.com/google/gofuzz"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/v2"
	scheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/cache"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
)

// Test that the discovery manager starts and aggregates from two local API services
func TestBasic(t *testing.T) {
	service1 := discoveryendpoint.NewResourceManager(scheme.Codecs)
	service2 := discoveryendpoint.NewResourceManager(scheme.Codecs)
	apiGroup1 := fuzzAPIGroups(2, 5, 25)
	apiGroup2 := fuzzAPIGroups(2, 5, 50)
	service1.SetGroups(apiGroup1.Groups)
	service2.SetGroups(apiGroup2.Groups)
	aggregatedResourceManager := discoveryendpoint.NewResourceManager(scheme.Codecs)
	aggregatedManager := NewDiscoveryManager(aggregatedResourceManager)

	for _, g := range apiGroup1.Groups {
		for _, v := range g.Versions {
			aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
				ObjectMeta: metav1.ObjectMeta{
					Name: v.Version + "." + g.Name,
				},
				Spec: apiregistrationv1.APIServiceSpec{
					Group:   g.Name,
					Version: v.Version,
					Service: &apiregistrationv1.ServiceReference{
						Name: "service1",
					},
				},
			}, service1)
		}
	}

	for _, g := range apiGroup2.Groups {
		for _, v := range g.Versions {
			aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
				ObjectMeta: metav1.ObjectMeta{
					Name: v.Version + "." + g.Name,
				},
				Spec: apiregistrationv1.APIServiceSpec{
					Group:   g.Name,
					Version: v.Version,
					Service: &apiregistrationv1.ServiceReference{
						Name: "service2",
					},
				},
			}, service2)
		}
	}

	container := restful.NewContainer()
	container.Add(aggregatedResourceManager.WebService())
	testCtx, _ := context.WithCancel(context.Background())
	go aggregatedManager.Run(testCtx.Done())

	cache.WaitForCacheSync(testCtx.Done(), aggregatedManager.ExternalServicesSynced)

	response, _, parsed := fetchPath(container, "/discovery/v2", "")
	if response.StatusCode != 200 {
		t.Fatalf("unexpected status code %d", response.StatusCode)
	}
	checkAPIGroups(t, apiGroup1, parsed)
	checkAPIGroups(t, apiGroup2, parsed)
}

func checkAPIGroups(t *testing.T, api metav1.APIGroupDiscoveryList, response *metav1.APIGroupDiscoveryList) {
	if len(response.Groups) < len(api.Groups) {
		t.Errorf("expected to check for at least %d groups, only have %d groups in response", len(api.Groups), len(response.Groups))
	}
	for _, knownGroup := range api.Groups {
		found := false
		for _, possibleGroup := range response.Groups {
			if knownGroup.Name == possibleGroup.Name {
				t.Logf("found %s", knownGroup.Name)
				found = true
			}
		}
		if found == false {
			t.Errorf("could not find %s", knownGroup.Name)
		}
	}
}

// Test that a handler associated with an APIService gets pinged after the
// APIService has been marked as dirty
func TestDirty(t *testing.T) {
	pinged := false
	service := discoveryendpoint.NewResourceManager(scheme.Codecs)
	aggregatedResourceManager := discoveryendpoint.NewResourceManager(scheme.Codecs)

	aggregatedManager := NewDiscoveryManager(aggregatedResourceManager)
	aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{
			Name: "v1.stable.example.com",
		},
		Spec: apiregistrationv1.APIServiceSpec{
			Group:   "stable.example.com",
			Version: "v1",
			Service: &apiregistrationv1.ServiceReference{
				Name: "test-service",
			},
		},
	}, http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		pinged = true
		service.ServeHTTP(w, r)
	}))
	// aggregatedManager.AddLocalDelegate()
	container := restful.NewContainer()
	container.Add(aggregatedResourceManager.WebService())
	testCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go aggregatedManager.Run(testCtx.Done())
	cache.WaitForCacheSync(testCtx.Done(), aggregatedManager.ExternalServicesSynced)

	// immediately check for ping, since Run() should block for local services
	if !pinged {
		t.Errorf("service handler never pinged")
	}
}

// Show that an APIService can be removed and that its group no longer remains
// if there are no versions
func TestRemoveAPIService(t *testing.T) {
	aggyService := discoveryendpoint.NewResourceManager(scheme.Codecs)
	service := discoveryendpoint.NewResourceManager(scheme.Codecs)
	apiGroup := fuzzAPIGroups(2, 3, 10)
	service.SetGroups(apiGroup.Groups)

	var apiServices []*apiregistrationv1.APIService
	for _, g := range apiGroup.Groups {
		for _, v := range g.Versions {
			apiservice := &apiregistrationv1.APIService{
				ObjectMeta: metav1.ObjectMeta{
					Name: v.Version + "." + g.Name,
				},
				Spec: apiregistrationv1.APIServiceSpec{
					Group:   g.Name,
					Version: v.Version,
					Service: &apiregistrationv1.ServiceReference{
						Namespace: "serviceNamespace",
						Name:      "serviceName",
					},
				},
			}

			apiServices = append(apiServices, apiservice)
		}
	}

	aggregatedManager := NewDiscoveryManager(aggyService)

	for _, s := range apiServices {
		aggregatedManager.AddAPIService(s, service)
	}

	container := restful.NewContainer()
	container.Add(aggyService.WebService())
	testCtx, _ := context.WithCancel(context.Background())
	go aggregatedManager.Run(testCtx.Done())

	for _, s := range apiServices {
		aggregatedManager.RemoveAPIService(s.Name)
	}

	cache.WaitForCacheSync(testCtx.Done(), aggregatedManager.ExternalServicesSynced)

	response, _, parsed := fetchPath(container, "/discovery/v2", "")
	if response.StatusCode != 200 {
		t.Fatalf("unexpected status code %d", response.StatusCode)
	}
	if len(parsed.Groups) > 0 {
		t.Errorf("expected to find no groups after service deletion (got %d groups)", len(parsed.Groups))
	}
}

// copied from staging/src/k8s.io/apiserver/pkg/endpoints/discovery/v2/handler_test.go
func fuzzAPIGroups(atLeastNumGroups, maxNumGroups int, seed int64) metav1.APIGroupDiscoveryList {
	fuzzer := fuzz.NewWithSeed(seed)
	fuzzer.NumElements(atLeastNumGroups, maxNumGroups)
	fuzzer.NilChance(0)
	fuzzer.Funcs(func(o *metav1.APIGroupDiscovery, c fuzz.Continue) {
		c.FuzzNoCustom(o)

		// The ResourceManager will just not serve the grouop if its versions
		// list is empty
		atLeastOne := metav1.APIVersionDiscovery{}
		c.Fuzz(&atLeastOne)
		o.Versions = append(o.Versions, atLeastOne)

		o.TypeMeta = metav1.TypeMeta{
			Kind:       "APIGroupDiscovery",
			APIVersion: "v1",
		}
	})

	var apis []metav1.APIGroupDiscovery
	fuzzer.Fuzz(&apis)

	return metav1.APIGroupDiscoveryList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "APIGroupDiscoveryList",
			APIVersion: "v1",
		},
		Groups: apis,
	}

}

// copied from staging/src/k8s.io/apiserver/pkg/endpoints/discovery/v2/handler_test.go
func fetchPath(handler http.Handler, path string, etag string) (*http.Response, []byte, *metav1.APIGroupDiscoveryList) {
	// Expect json-formatted apis group list
	w := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/discovery/v2", nil)

	// Ask for JSON response
	req.Header.Set("Accept", runtime.ContentTypeProtobuf)

	if etag != "" {
		// Quote provided etag if unquoted
		quoted := etag
		if !strings.HasPrefix(etag, "\"") {
			quoted = strconv.Quote(etag)
		}
		req.Header.Set("If-None-Match", quoted)
	}

	handler.ServeHTTP(w, req)

	bytes := w.Body.Bytes()
	var decoded *metav1.APIGroupDiscoveryList
	if len(bytes) > 0 {
		decoded = &metav1.APIGroupDiscoveryList{}
		runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), bytes, decoded)
	}

	return w.Result(), bytes, decoded
}
