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
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"
	"github.com/stretchr/testify/require"

	apidiscoveryv2beta1 "k8s.io/api/apidiscovery/v2beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	discoveryendpoint "k8s.io/apiserver/pkg/endpoints/discovery/aggregated"
	scheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	apiregistrationv1 "k8s.io/kube-aggregator/pkg/apis/apiregistration/v1"
)

func newDiscoveryManager(rm discoveryendpoint.ResourceManager) *discoveryManager {
	dm := NewDiscoveryManager(rm).(*discoveryManager)
	dm.dirtyAPIServiceQueue = newCompleterWorkqueue(dm.dirtyAPIServiceQueue)

	return dm
}

// Returns true if the queue of services to sync is complete which means
// everything has been reconciled and placed into merged document
func waitForQueueComplete(stopCh <-chan struct{}, dm *discoveryManager) bool {
	return cache.WaitForCacheSync(stopCh, func() bool {
		return dm.dirtyAPIServiceQueue.(*completerWorkqueue).isComplete()
	})
}

func fakeCh() chan struct{} {
	return make(chan struct{})
}

// Test that the discovery manager starts and aggregates from two local API services
func TestBasic(t *testing.T) {
	service1 := discoveryendpoint.NewResourceManager("apis")
	service2 := discoveryendpoint.NewResourceManager("apis")
	service3 := discoveryendpoint.NewResourceManager("apis")
	apiGroup1 := fuzzAPIGroups(2, 5, 25)
	apiGroup2 := fuzzAPIGroups(2, 5, 50)
	apiGroup3 := apidiscoveryv2beta1.APIGroupDiscoveryList{Items: []apidiscoveryv2beta1.APIGroupDiscovery{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "weird.example.com"},
			Versions: []apidiscoveryv2beta1.APIVersionDiscovery{
				{
					Version:   "v1",
					Freshness: "Current",
					Resources: []apidiscoveryv2beta1.APIResourceDiscovery{
						{
							Resource: "parent-missing-kind",
							Subresources: []apidiscoveryv2beta1.APISubresourceDiscovery{
								{Subresource: "subresource-missing-kind"},
							},
						},
						{
							Resource:     "parent-empty-kind",
							ResponseKind: &metav1.GroupVersionKind{},
							Subresources: []apidiscoveryv2beta1.APISubresourceDiscovery{
								{Subresource: "subresource-empty-kind", ResponseKind: &metav1.GroupVersionKind{}},
							},
						},
						{
							Resource:     "parent-with-kind",
							ResponseKind: &metav1.GroupVersionKind{Kind: "ParentWithKind"},
							Subresources: []apidiscoveryv2beta1.APISubresourceDiscovery{
								{Subresource: "subresource-with-kind", ResponseKind: &metav1.GroupVersionKind{Kind: "SubresourceWithKind"}},
							},
						},
					},
				},
			},
		},
	}}
	apiGroup3WithFixup := apidiscoveryv2beta1.APIGroupDiscoveryList{Items: []apidiscoveryv2beta1.APIGroupDiscovery{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "weird.example.com"},
			Versions: []apidiscoveryv2beta1.APIVersionDiscovery{
				{
					Version:   "v1",
					Freshness: "Current",
					Resources: []apidiscoveryv2beta1.APIResourceDiscovery{
						{
							Resource:     "parent-missing-kind",
							ResponseKind: &metav1.GroupVersionKind{}, // defaulted by aggregator
							Subresources: []apidiscoveryv2beta1.APISubresourceDiscovery{
								{Subresource: "subresource-missing-kind", ResponseKind: &metav1.GroupVersionKind{}}, // defaulted by aggregator
							},
						},
						{
							Resource:     "parent-empty-kind",
							ResponseKind: &metav1.GroupVersionKind{},
							Subresources: []apidiscoveryv2beta1.APISubresourceDiscovery{
								{Subresource: "subresource-empty-kind", ResponseKind: &metav1.GroupVersionKind{}},
							},
						},
						{
							Resource:     "parent-with-kind",
							ResponseKind: &metav1.GroupVersionKind{Kind: "ParentWithKind"},
							Subresources: []apidiscoveryv2beta1.APISubresourceDiscovery{
								{Subresource: "subresource-with-kind", ResponseKind: &metav1.GroupVersionKind{Kind: "SubresourceWithKind"}},
							},
						},
					},
				},
			},
		},
	}}
	service1.SetGroups(apiGroup1.Items)
	service2.SetGroups(apiGroup2.Items)
	service3.SetGroups(apiGroup3.Items)
	aggregatedResourceManager := discoveryendpoint.NewResourceManager("apis")
	aggregatedManager := newDiscoveryManager(aggregatedResourceManager)

	for _, g := range apiGroup1.Items {
		versionPriority := int32(len(g.Versions) + 1)
		for _, v := range g.Versions {
			versionPriority--
			aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
				ObjectMeta: metav1.ObjectMeta{
					Name: v.Version + "." + g.Name,
				},
				Spec: apiregistrationv1.APIServiceSpec{
					Group:           g.Name,
					Version:         v.Version,
					VersionPriority: versionPriority,
					Service: &apiregistrationv1.ServiceReference{
						Name: "service1",
					},
				},
			}, service1)
		}
	}

	for _, g := range apiGroup2.Items {
		versionPriority := int32(len(g.Versions) + 1)
		for _, v := range g.Versions {
			versionPriority--
			aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
				ObjectMeta: metav1.ObjectMeta{
					Name: v.Version + "." + g.Name,
				},
				Spec: apiregistrationv1.APIServiceSpec{
					Group:           g.Name,
					Version:         v.Version,
					VersionPriority: versionPriority,
					Service: &apiregistrationv1.ServiceReference{
						Name: "service2",
					},
				},
			}, service2)
		}
	}

	for _, g := range apiGroup3.Items {
		versionPriority := int32(len(g.Versions) + 1)
		for _, v := range g.Versions {
			versionPriority--
			aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
				ObjectMeta: metav1.ObjectMeta{
					Name: v.Version + "." + g.Name,
				},
				Spec: apiregistrationv1.APIServiceSpec{
					Group:           g.Name,
					Version:         v.Version,
					VersionPriority: versionPriority,
					Service: &apiregistrationv1.ServiceReference{
						Name: "service3",
					},
				},
			}, service3)
		}
	}

	testCtx, testCancel := context.WithCancel(context.Background())
	defer testCancel()

	go aggregatedManager.Run(testCtx.Done(), fakeCh())

	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))

	response, _, parsed := fetchPath(aggregatedResourceManager, "")
	if response.StatusCode != 200 {
		t.Fatalf("unexpected status code %d", response.StatusCode)
	}
	checkAPIGroups(t, apiGroup1, parsed)
	checkAPIGroups(t, apiGroup2, parsed)
	checkAPIGroups(t, apiGroup3WithFixup, parsed)
}

func checkAPIGroups(t *testing.T, api apidiscoveryv2beta1.APIGroupDiscoveryList, response *apidiscoveryv2beta1.APIGroupDiscoveryList) {
	t.Helper()
	if len(response.Items) < len(api.Items) {
		t.Errorf("expected to check for at least %d groups, only have %d groups in response", len(api.Items), len(response.Items))
	}
	for _, knownGroup := range api.Items {
		found := false
		for _, possibleGroup := range response.Items {
			if knownGroup.Name == possibleGroup.Name {
				t.Logf("found %s", knownGroup.Name)
				found = true
				diff := cmp.Diff(knownGroup, possibleGroup)
				if len(diff) > 0 {
					t.Error(diff)
				}
			}
		}
		if found == false {
			t.Errorf("could not find %s", knownGroup.Name)
		}
	}
}

// TestInitialRunHasAllAPIServices tests that when discovery is ready, all APIService
// are present and ones that have not synced are in the list as Stale.
func TestInitialRunHasAllAPIServices(t *testing.T) {
	neverReturnCh := make(chan struct{})
	defer close(neverReturnCh)
	service := discoveryendpoint.NewResourceManager("apis")
	aggregatedResourceManager := discoveryendpoint.NewResourceManager("apis")

	aggregatedManager := newDiscoveryManager(aggregatedResourceManager)

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
		<-neverReturnCh
		service.ServeHTTP(w, r)
	}))
	testCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	initialSyncedCh := fakeCh()
	go aggregatedManager.Run(testCtx.Done(), initialSyncedCh)
	select {
	case <-initialSyncedCh:
	case <-time.After(10 * time.Second):
		t.Fatal("timed out waiting for initial sync")
	}

	response, _, parsed := fetchPath(aggregatedResourceManager, "")
	if response.StatusCode != 200 {
		t.Fatalf("unexpected status code %d", response.StatusCode)
	}

	apiGroup := apidiscoveryv2beta1.APIGroupDiscoveryList{Items: []apidiscoveryv2beta1.APIGroupDiscovery{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "stable.example.com"},
			Versions: []apidiscoveryv2beta1.APIVersionDiscovery{
				{
					Version:   "v1",
					Freshness: "Stale",
				},
			},
		},
	}}

	checkAPIGroups(t, apiGroup, parsed)
}

// Test that a handler associated with an APIService gets pinged after the
// APIService has been marked as dirty
func TestDirty(t *testing.T) {
	var pinged atomic.Bool
	service := discoveryendpoint.NewResourceManager("apis")
	aggregatedResourceManager := discoveryendpoint.NewResourceManager("apis")

	aggregatedManager := newDiscoveryManager(aggregatedResourceManager)

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
		pinged.Store(true)
		service.ServeHTTP(w, r)
	}))
	testCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go aggregatedManager.Run(testCtx.Done(), fakeCh())
	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))

	// immediately check for ping, since Run() should block for local services
	if !pinged.Load() {
		t.Errorf("service handler never pinged")
	}
}

// Shows that waitForQueueComplete also waits for syncing to
// complete by artificially making the sync handler take a long time
func TestWaitForSync(t *testing.T) {
	pinged := atomic.Bool{}
	service := discoveryendpoint.NewResourceManager("apis")
	aggregatedResourceManager := discoveryendpoint.NewResourceManager("apis")

	aggregatedManager := newDiscoveryManager(aggregatedResourceManager)

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
		time.Sleep(3 * time.Second)
		pinged.Store(true)
		service.ServeHTTP(w, r)
	}))
	testCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go aggregatedManager.Run(testCtx.Done(), fakeCh())
	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))

	// immediately check for ping, since Run() should block for local services
	if !pinged.Load() {
		t.Errorf("service handler never pinged")
	}
}

// Show that an APIService can be removed and that its group no longer remains
// if there are no versions
func TestRemoveAPIService(t *testing.T) {
	aggyService := discoveryendpoint.NewResourceManager("apis")
	service := discoveryendpoint.NewResourceManager("apis")
	apiGroup := fuzzAPIGroups(2, 3, 10)
	service.SetGroups(apiGroup.Items)

	var apiServices []*apiregistrationv1.APIService
	for _, g := range apiGroup.Items {
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

	aggregatedManager := newDiscoveryManager(aggyService)

	for _, s := range apiServices {
		aggregatedManager.AddAPIService(s, service)
	}

	testCtx, testCancel := context.WithCancel(context.Background())
	defer testCancel()

	go aggregatedManager.Run(testCtx.Done(), fakeCh())

	for _, s := range apiServices {
		aggregatedManager.RemoveAPIService(s.Name)
	}

	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))

	response, _, parsed := fetchPath(aggyService, "")
	if response.StatusCode != 200 {
		t.Fatalf("unexpected status code %d", response.StatusCode)
	}
	if len(parsed.Items) > 0 {
		t.Errorf("expected to find no groups after service deletion (got %d groups)", len(parsed.Items))
	}
}

func TestLegacyFallbackNoCache(t *testing.T) {
	aggregatedResourceManager := discoveryendpoint.NewResourceManager("apis")
	rootAPIsHandler := discovery.NewRootAPIsHandler(discovery.DefaultAddresses{DefaultAddress: "192.168.1.1"}, scheme.Codecs)

	legacyGroupHandler := discovery.NewAPIGroupHandler(scheme.Codecs, metav1.APIGroup{
		Name: "stable.example.com",
		PreferredVersion: metav1.GroupVersionForDiscovery{
			GroupVersion: "stable.example.com/v1",
			Version:      "v1",
		},
		Versions: []metav1.GroupVersionForDiscovery{
			{
				GroupVersion: "stable.example.com/v1",
				Version:      "v1",
			},
			{
				GroupVersion: "stable.example.com/v1beta1",
				Version:      "v1beta1",
			},
			{
				GroupVersion: "stable.example.com/v1alpha1",
				Version:      "v1alpha1",
			},
			{
				GroupVersion: "stable.example.com/v2alpha1",
				Version:      "v2alpha1",
			},
		},
	})

	generateVersionResource := func(version string) metav1.APIResource {
		return metav1.APIResource{
			Name:         "foos",
			SingularName: "foo",
			Group:        "stable.example.com",
			Version:      version,
			Namespaced:   false,
			Kind:         "Foo",
			Verbs:        []string{"get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"},
			Categories:   []string{"all"},
		}
	}

	resources := map[string]metav1.APIResource{
		"v1":       generateVersionResource("v1"),
		"v1beta1":  generateVersionResource("v1beta1"),
		"v1alpha1": generateVersionResource("v1alpha1"),
	}

	legacyResourceHandlerV1 := discovery.NewAPIVersionHandler(scheme.Codecs, schema.GroupVersion{
		Group:   "stable.example.com",
		Version: "v1",
	}, discovery.APIResourceListerFunc(func() []metav1.APIResource {
		return []metav1.APIResource{
			resources["v1"],
		}
	}))

	legacyResourceHandlerV1Beta1 := discovery.NewAPIVersionHandler(scheme.Codecs, schema.GroupVersion{
		Group:   "stable.example.com",
		Version: "v1beta1",
	}, discovery.APIResourceListerFunc(func() []metav1.APIResource {
		return []metav1.APIResource{
			resources["v1beta1"],
		}
	}))

	legacyResourceHandlerV1Alpha1 := discovery.NewAPIVersionHandler(scheme.Codecs, schema.GroupVersion{
		Group:   "stable.example.com",
		Version: "v1alpha1",
	}, discovery.APIResourceListerFunc(func() []metav1.APIResource {
		return []metav1.APIResource{
			resources["v1alpha1"],
		}
	}))

	handlerFunc := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/apis/stable.example.com" {
			legacyGroupHandler.ServeHTTP(w, r)
		} else if r.URL.Path == "/apis/stable.example.com/v1" {
			// defer to legacy discovery
			legacyResourceHandlerV1.ServeHTTP(w, r)
		} else if r.URL.Path == "/apis/stable.example.com/v1beta1" {
			// defer to legacy discovery
			legacyResourceHandlerV1Beta1.ServeHTTP(w, r)
		} else if r.URL.Path == "/apis/stable.example.com/v1alpha1" {
			legacyResourceHandlerV1Alpha1.ServeHTTP(w, r)
		} else if r.URL.Path == "/apis/stable.example.com/v2alpha1" {
			// serve the most minimal discovery doc that could have worked prior to aggregated discovery
			json.NewEncoder(w).Encode(&metav1.APIResourceList{
				GroupVersion: "stable.example.com/v2alpha1",
				APIResources: []metav1.APIResource{
					{Name: "parent-without-kind"},
					{Name: "missing-parent/subresource-without-parent", Kind: "SubresourceWithoutParent"},
					{Name: "parent-without-kind/subresource", Kind: "Subresource"},
					{Name: "parent-without-kind/subresource-without-kind"},
				},
			})
		} else if r.URL.Path == "/apis" {
			rootAPIsHandler.ServeHTTP(w, r)
		} else {
			// Unknown url
			t.Fatalf("unexpected request sent to %v", r.URL.Path)
		}
	})

	aggregatedManager := newDiscoveryManager(aggregatedResourceManager)
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
	}, handlerFunc)
	aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{
			Name: "v1beta1.stable.example.com",
		},
		Spec: apiregistrationv1.APIServiceSpec{
			Group:   "stable.example.com",
			Version: "v1beta1",
			Service: &apiregistrationv1.ServiceReference{
				Name: "test-service",
			},
		},
	}, handlerFunc)
	aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{
			Name: "v1alpha1.stable.example.com",
		},
		Spec: apiregistrationv1.APIServiceSpec{
			Group:   "stable.example.com",
			Version: "v1alpha1",
			Service: &apiregistrationv1.ServiceReference{
				Name: "test-service",
			},
		},
	}, handlerFunc)
	aggregatedManager.AddAPIService(&apiregistrationv1.APIService{
		ObjectMeta: metav1.ObjectMeta{
			Name: "v2alpha1.stable.example.com",
		},
		Spec: apiregistrationv1.APIServiceSpec{
			Group:   "stable.example.com",
			Version: "v2alpha1",
			Service: &apiregistrationv1.ServiceReference{
				Name: "test-service",
			},
		},
	}, handlerFunc)

	testCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go aggregatedManager.Run(testCtx.Done(), fakeCh())
	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))

	// At this point external services have synced. Check if discovery document
	// includes the legacy resources
	_, _, doc := fetchPath(aggregatedResourceManager, "")

	mustConvert := func(r []metav1.APIResource) []apidiscoveryv2beta1.APIResourceDiscovery {
		converted, err := endpoints.ConvertGroupVersionIntoToDiscovery(r)
		require.NoError(t, err)
		return converted
	}
	expectAggregatedDiscovery := []apidiscoveryv2beta1.APIGroupDiscovery{{
		ObjectMeta: metav1.ObjectMeta{
			Name: "stable.example.com",
		},
		Versions: []apidiscoveryv2beta1.APIVersionDiscovery{
			{
				Version:   "v1",
				Resources: mustConvert([]metav1.APIResource{resources["v1"]}),
				Freshness: apidiscoveryv2beta1.DiscoveryFreshnessCurrent,
			},
			{
				Version:   "v1beta1",
				Resources: mustConvert([]metav1.APIResource{resources["v1beta1"]}),
				Freshness: apidiscoveryv2beta1.DiscoveryFreshnessCurrent,
			},
			{
				Version: "v2alpha1",
				Resources: []apidiscoveryv2beta1.APIResourceDiscovery{
					{
						Resource:     "parent-without-kind",
						ResponseKind: &metav1.GroupVersionKind{}, // defaulted
						Scope:        "Cluster",
						Subresources: []apidiscoveryv2beta1.APISubresourceDiscovery{
							{
								Subresource:  "subresource",
								ResponseKind: &metav1.GroupVersionKind{Kind: "Subresource"},
							},
							{
								Subresource:  "subresource-without-kind",
								ResponseKind: &metav1.GroupVersionKind{}, // defaulted
							},
						},
					},
					{
						Resource:     "missing-parent",
						ResponseKind: &metav1.GroupVersionKind{}, // defaulted
						Scope:        "Cluster",
						Subresources: []apidiscoveryv2beta1.APISubresourceDiscovery{
							{
								Subresource:  "subresource-without-parent",
								ResponseKind: &metav1.GroupVersionKind{Kind: "SubresourceWithoutParent"},
							},
						},
					},
				},
				Freshness: apidiscoveryv2beta1.DiscoveryFreshnessCurrent,
			},
			{
				Version:   "v1alpha1",
				Resources: mustConvert([]metav1.APIResource{resources["v1alpha1"]}),
				Freshness: apidiscoveryv2beta1.DiscoveryFreshnessCurrent,
			},
		},
	}}
	require.Equal(t, doc.Items, expectAggregatedDiscovery)
}

func testLegacyFallbackWithCustomRootHandler(t *testing.T, rootHandlerFn func(http.ResponseWriter, *http.Request)) {
	aggregatedResourceManager := discoveryendpoint.NewResourceManager("apis")

	legacyGroupHandler := discovery.NewAPIGroupHandler(scheme.Codecs, metav1.APIGroup{
		Name: "stable.example.com",
		PreferredVersion: metav1.GroupVersionForDiscovery{
			GroupVersion: "stable.example.com/v1",
			Version:      "v1",
		},
		Versions: []metav1.GroupVersionForDiscovery{
			{
				GroupVersion: "stable.example.com/v1",
				Version:      "v1",
			},
			{
				GroupVersion: "stable.example.com/v1beta1",
				Version:      "v1beta1",
			},
		},
	})

	resource := metav1.APIResource{
		Name:         "foos",
		SingularName: "foo",
		Group:        "stable.example.com",
		Version:      "v1",
		Namespaced:   false,
		Kind:         "Foo",
		Verbs:        []string{"get", "list", "watch", "create", "update", "patch", "delete", "deletecollection"},
		Categories:   []string{"all"},
	}

	legacyResourceHandler := discovery.NewAPIVersionHandler(scheme.Codecs, schema.GroupVersion{
		Group:   "stable.example.com",
		Version: "v1",
	}, discovery.APIResourceListerFunc(func() []metav1.APIResource {
		return []metav1.APIResource{
			resource,
		}
	}))

	aggregatedManager := newDiscoveryManager(aggregatedResourceManager)
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
		if r.URL.Path == "/apis/stable.example.com" {
			legacyGroupHandler.ServeHTTP(w, r)
		} else if r.URL.Path == "/apis/stable.example.com/v1" {
			// defer to legacy discovery
			legacyResourceHandler.ServeHTTP(w, r)
		} else if r.URL.Path == "/apis" {
			rootHandlerFn(w, r)
		} else {
			// Unknown url
			t.Fatalf("unexpected request sent to %v", r.URL.Path)
		}
	}))
	testCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go aggregatedManager.Run(testCtx.Done(), fakeCh())
	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))

	// At this point external services have synced. Check if discovery document
	// includes the legacy resources
	_, _, doc := fetchPath(aggregatedResourceManager, "")

	converted, err := endpoints.ConvertGroupVersionIntoToDiscovery([]metav1.APIResource{resource})
	require.NoError(t, err)
	require.Equal(t, []apidiscoveryv2beta1.APIGroupDiscovery{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: resource.Group,
			},
			Versions: []apidiscoveryv2beta1.APIVersionDiscovery{
				{
					Version:   resource.Version,
					Resources: converted,
					Freshness: apidiscoveryv2beta1.DiscoveryFreshnessCurrent,
				},
			},
		},
	}, doc.Items)
}
func TestLegacyFallback(t *testing.T) {
	rootAPIsHandler := discovery.NewRootAPIsHandler(discovery.DefaultAddresses{DefaultAddress: "192.168.1.1"}, scheme.Codecs)
	testCases := []struct {
		name        string
		rootHandler func(http.ResponseWriter, *http.Request)
	}{
		{
			name:        "Default root handler (406)",
			rootHandler: rootAPIsHandler.ServeHTTP,
		},
		{
			name: "Root handler with non 200 status code",
			rootHandler: func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(404)
			},
		},
		{
			name: "Root handler with 200 response code no content type",
			rootHandler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(200)
			},
		},
		{
			name: "Root handler with 200 response code incorrect content type",
			rootHandler: func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json;g=apidiscovery.k8s.io;v=v1alpha1;as=APIGroupDiscoveryList")
				w.WriteHeader(200)
			},
		},
	}
	for _, tc := range testCases {
		testLegacyFallbackWithCustomRootHandler(t, tc.rootHandler)
	}
}

func TestAPIServiceStale(t *testing.T) {
	aggregatedResourceManager := discoveryendpoint.NewResourceManager("apis")
	aggregatedManager := newDiscoveryManager(aggregatedResourceManager)
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
		w.WriteHeader(503)
	}))
	testCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go aggregatedManager.Run(testCtx.Done(), fakeCh())
	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))

	// At this point external services have synced. Check if discovery document
	// lists the APIService group version as Stale.
	_, _, doc := fetchPath(aggregatedResourceManager, "")
	require.Equal(t, []apidiscoveryv2beta1.APIGroupDiscovery{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "stable.example.com",
			},
			Versions: []apidiscoveryv2beta1.APIVersionDiscovery{
				{
					Version:   "v1",
					Freshness: apidiscoveryv2beta1.DiscoveryFreshnessStale,
				},
			},
		},
	}, doc.Items)
}

// Exercises the 304 Not Modified Path of the aggregator
// This path in 1.26.0 would result in a deadlock if an aggregated APIService
// returned a 304 Not Modified response for its own aggregated discovery document.
func TestNotModified(t *testing.T) {
	aggyService := discoveryendpoint.NewResourceManager("apis")
	service := discoveryendpoint.NewResourceManager("apis")
	apiGroup := fuzzAPIGroups(2, 3, 10)
	service.SetGroups(apiGroup.Items)

	var apiServices []*apiregistrationv1.APIService
	for _, g := range apiGroup.Items {
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

	aggregatedManager := newDiscoveryManager(aggyService)
	// Add all except the last group.
	// Ensure this is done BEFORE the call to run, so they are included in initial
	// list to keep test focused
	for _, s := range apiServices[:len(apiServices)-1] {
		aggregatedManager.AddAPIService(s, service)
	}

	testCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	go aggregatedManager.Run(testCtx.Done(), fakeCh())

	// Important to wait here to ensure we prime the cache with the initial list
	// of documents in order to exercise 304 Not Modified
	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))

	// Now add all groups. We excluded one group before so that waitForQueueIsComplete
	// could include it in this round. Now, if waitForQueueIsComplete ever returns
	// true, it must have synced all the pre-existing groups before, which would
	// return 304 Not Modified
	for _, s := range apiServices {
		aggregatedManager.AddAPIService(s, service)
	}

	// This would wait the full timeout on 1.26.0.
	require.True(t, waitForQueueComplete(testCtx.Done(), aggregatedManager))
}

// copied from staging/src/k8s.io/apiserver/pkg/endpoints/discovery/v2/handler_test.go
func fuzzAPIGroups(atLeastNumGroups, maxNumGroups int, seed int64) apidiscoveryv2beta1.APIGroupDiscoveryList {
	fuzzer := fuzz.NewWithSeed(seed)
	fuzzer.NumElements(atLeastNumGroups, maxNumGroups)
	fuzzer.NilChance(0)
	fuzzer.Funcs(func(o *apidiscoveryv2beta1.APIGroupDiscovery, c fuzz.Continue) {
		c.FuzzNoCustom(o)

		// The ResourceManager will just not serve the group if its versions
		// list is empty
		atLeastOne := apidiscoveryv2beta1.APIVersionDiscovery{}
		c.Fuzz(&atLeastOne)
		o.Versions = append(o.Versions, atLeastOne)

		// clear invalid fuzzed values
		o.TypeMeta = metav1.TypeMeta{}
		// truncate object meta to just name
		o.ObjectMeta = metav1.ObjectMeta{Name: o.ObjectMeta.Name}
		// fix version freshness value, make versions unique and non-empty
		for i := range o.Versions {
			o.Versions[i].Freshness = "Current"
			o.Versions[i].Version = fmt.Sprintf("v%d", i+1)
		}
	})

	var apis []apidiscoveryv2beta1.APIGroupDiscovery
	fuzzer.Fuzz(&apis)

	return apidiscoveryv2beta1.APIGroupDiscoveryList{
		TypeMeta: metav1.TypeMeta{
			Kind:       "APIGroupDiscoveryList",
			APIVersion: "v1",
		},
		Items: apis,
	}

}

// copied from staging/src/k8s.io/apiserver/pkg/endpoints/discovery/v2/handler_test.go
func fetchPath(handler http.Handler, etag string) (*http.Response, []byte, *apidiscoveryv2beta1.APIGroupDiscoveryList) {
	// Expect json-formatted apis group list
	w := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/apis", nil)

	// Ask for JSON response
	req.Header.Set("Accept", runtime.ContentTypeJSON+";g=apidiscovery.k8s.io;v=v2beta1;as=APIGroupDiscoveryList")

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
	var decoded *apidiscoveryv2beta1.APIGroupDiscoveryList
	if len(bytes) > 0 {
		decoded = &apidiscoveryv2beta1.APIGroupDiscoveryList{}
		runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), bytes, decoded)
	}

	return w.Result(), bytes, decoded
}

// completerWorkqueue is a workqueue.RateLimitingInterface that implements
// isComplete
type completerWorkqueue struct {
	lock sync.Mutex
	workqueue.RateLimitingInterface
	processing map[interface{}]struct{}
}

var _ = workqueue.RateLimitingInterface(&completerWorkqueue{})

func newCompleterWorkqueue(wq workqueue.RateLimitingInterface) *completerWorkqueue {
	return &completerWorkqueue{
		RateLimitingInterface: wq,
		processing:            make(map[interface{}]struct{}),
	}
}

func (q *completerWorkqueue) Add(item interface{}) {
	q.lock.Lock()
	defer q.lock.Unlock()
	q.processing[item] = struct{}{}
	q.RateLimitingInterface.Add(item)
}

func (q *completerWorkqueue) AddAfter(item interface{}, duration time.Duration) {
	q.Add(item)
}

func (q *completerWorkqueue) AddRateLimited(item interface{}) {
	q.Add(item)
}

func (q *completerWorkqueue) Done(item interface{}) {
	q.lock.Lock()
	defer q.lock.Unlock()
	delete(q.processing, item)
	q.RateLimitingInterface.Done(item)
}

func (q *completerWorkqueue) isComplete() bool {
	q.lock.Lock()
	defer q.lock.Unlock()
	return q.Len() == 0 && len(q.processing) == 0
}
