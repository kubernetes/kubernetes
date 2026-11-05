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

package rest

import (
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/server/storage"
	apiserverstorage "k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	podregistry "k8s.io/kubernetes/pkg/registry/core/pod"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
)

func TestGetServersToValidate(t *testing.T) {
	servers := componentStatusStorage{fakeStorageFactory{}}.serversToValidate()

	if e, a := 3, len(servers); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	for _, server := range []string{"scheduler", "controller-manager", "etcd-0"} {
		if _, ok := servers[server]; !ok {
			t.Errorf("server list missing: %s", server)
		}
	}
}

type fakeStorageFactory struct{}

func (f fakeStorageFactory) NewConfig(groupResource schema.GroupResource, example runtime.Object) (*storagebackend.ConfigForResource, error) {
	return nil, nil
}

func (f fakeStorageFactory) ResourcePrefix(groupResource schema.GroupResource) string {
	return ""
}

func (f fakeStorageFactory) Backends() []storage.Backend {
	return []storage.Backend{{Server: "etcd-0"}}
}

func (f fakeStorageFactory) Configs() []storagebackend.Config {
	return []storagebackend.Config{{Transport: storagebackend.TransportConfig{ServerList: []string{"etcd-0"}}}}
}

func newTestCoreStorage(t *testing.T, disabledPodSubresources sets.Set[string]) map[string]rest.Storage {
	t.Helper()

	nodePortAllocator, err := portallocator.NewInMemory(utilnet.PortRange{Base: 30000, Size: 100})
	if err != nil {
		t.Fatalf("portallocator.NewInMemory failed: %v", err)
	}

	provider := &legacyProvider{
		Config: Config{
			GenericConfig: GenericConfig{
				StorageFactory:       fakeStorageFactory{},
				LoopbackClientConfig: &restclient.Config{},
			},
			DisabledPodSubresources: disabledPodSubresources,
		},
		serviceNodePortAllocator: nodePortAllocator,
	}

	resourceConfig := storage.NewResourceConfig()
	resourceConfig.EnableVersions(corev1.SchemeGroupVersion)

	restOptions := generic.RESTOptions{
		StorageConfig: &storagebackend.ConfigForResource{},
		Decorator: func(
			*storagebackend.ConfigForResource,
			string,
			generic.CacheKeyFunc,
			apiserverstorage.ReverseKeyFunc,
			func() runtime.Object,
			func() runtime.Object,
			apiserverstorage.AttrFunc,
			apiserverstorage.IndexerFuncs,
			*cache.Indexers,
		) (apiserverstorage.Interface, factory.DestroyFunc, error) {
			return nil, func() {}, nil
		},
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "registry",
	}

	apiGroupInfo, err := provider.NewRESTStorage(resourceConfig, restOptions)
	if err != nil {
		t.Fatalf("NewRESTStorage failed: %v", err)
	}
	t.Cleanup(func() {
		for _, stores := range apiGroupInfo.VersionedResourcesStorageMap {
			for _, store := range stores {
				if store != nil {
					store.Destroy()
				}
			}
		}
	})

	return apiGroupInfo.VersionedResourcesStorageMap["v1"]
}

func podSubresourcesFromStorage(t *testing.T, v1Storage map[string]rest.Storage) sets.Set[string] {
	t.Helper()

	got := sets.New[string]()
	for key, store := range v1Storage {
		if subresource, ok := strings.CutPrefix(key, "pods/"); ok {
			if store == nil {
				t.Errorf("storage entry %q was registered with a nil rest.Storage", key)
			}
			got.Insert(subresource)
		}
	}
	return got
}

func TestPodSubresourceClassification(t *testing.T) {
	if overlap := podregistry.DisableablePodSubresources().Intersection(podregistry.NonDisableablePodSubresources()); overlap.Len() > 0 {
		t.Errorf("DisableablePodSubresources() and NonDisableablePodSubresources() overlap: %v", sets.List(overlap))
	}

	// Enable gated pod subresources so any newly added pods/<subresource> endpoint is forced
	// to classify itself in DisableablePodSubresources or NonDisableablePodSubresources.
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, true)
	got := podSubresourcesFromStorage(t, newTestCoreStorage(t, nil))
	want := podregistry.DisableablePodSubresources().Union(podregistry.NonDisableablePodSubresources())
	if !got.Equal(want) {
		t.Errorf("registered pods/<subresource> keys do not match classified sets:\n  unclassified: %v\n  unregistered: %v",
			sets.List(got.Difference(want)),
			sets.List(want.Difference(got)),
		)
	}
}

func TestDisabledPodSubresources(t *testing.T) {
	allSubresources := podregistry.DisableablePodSubresources().Union(podregistry.NonDisableablePodSubresources())

	testCases := []struct {
		name              string
		resizeFeatureGate bool
		disabled          sets.Set[string]
		wantSubresources  sets.Set[string]
	}{
		{
			name:              "subset of disableable subresources",
			resizeFeatureGate: true,
			disabled:          sets.New("ephemeralcontainers", "eviction", "exec", "proxy", "resize"),
			wantSubresources:  allSubresources.Clone().Delete("ephemeralcontainers", "eviction", "exec", "proxy", "resize"),
		},
		{
			name:              "all disableable subresources",
			resizeFeatureGate: true,
			disabled:          podregistry.DisableablePodSubresources(),
			wantSubresources:  podregistry.NonDisableablePodSubresources(),
		},
		{
			name:              "resize feature gate disabled",
			resizeFeatureGate: false,
			disabled:          sets.New("log", "resize"),
			wantSubresources:  allSubresources.Clone().Delete("log", "resize"),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if !tc.resizeFeatureGate {
				// InPlacePodVerticalScaling is GA and locked to true at 1.35+, so emulate 1.34 to exercise the gate-off branch.
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.34"))
			}
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.InPlacePodVerticalScaling, tc.resizeFeatureGate)

			v1Storage := newTestCoreStorage(t, tc.disabled)
			for _, resource := range []string{"pods", "bindings"} {
				if v1Storage[resource] == nil {
					t.Errorf("expected %q to remain registered with non-nil storage", resource)
				}
			}

			got := podSubresourcesFromStorage(t, v1Storage)
			if !got.Equal(tc.wantSubresources) {
				t.Errorf("pods/<subresource> keys = %v, want %v (missing=%v, unexpected=%v)",
					sets.List(got),
					sets.List(tc.wantSubresources),
					sets.List(tc.wantSubresources.Difference(got)),
					sets.List(got.Difference(tc.wantSubresources)),
				)
			}
		})
	}
}
