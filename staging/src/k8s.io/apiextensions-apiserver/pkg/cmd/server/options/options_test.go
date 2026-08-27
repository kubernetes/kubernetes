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

package options

import (
	"reflect"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	genericregistry "k8s.io/apiserver/pkg/registry/generic"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	flowcontrolrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
)

func TestNewCRDRESTOptionsGetterWatchCacheSizes(t *testing.T) {
	testCases := []struct {
		name                 string
		enableWatchCache     bool
		watchCacheSizes      []string
		resource             schema.GroupResource
		expectStorageCacher  bool
		expectUndecorated    bool
	}{
		{
			name:                "watch cache enabled, watch-cache-sizes has #0 for this CRD: should still use watch cache",
			enableWatchCache:    true,
			watchCacheSizes:     []string{"crontabs.stable.example.com#0"},
			resource:            schema.GroupResource{Group: "stable.example.com", Resource: "crontabs"},
			expectStorageCacher: true,
			expectUndecorated:   false,
		},
		{
			name:                "watch cache enabled, watch-cache-sizes has #0 for multiple CRDs: should still use watch cache",
			enableWatchCache:    true,
			watchCacheSizes:     []string{"crontabs.stable.example.com#0", "foos.custom.example.com#0"},
			resource:            schema.GroupResource{Group: "custom.example.com", Resource: "foos"},
			expectStorageCacher: true,
			expectUndecorated:   false,
		},
		{
			name:                "watch cache enabled, watch-cache-sizes is nil: should use watch cache",
			enableWatchCache:    true,
			watchCacheSizes:     nil,
			resource:            schema.GroupResource{Group: "stable.example.com", Resource: "crontabs"},
			expectStorageCacher: true,
			expectUndecorated:   false,
		},
		{
			name:                "watch cache disabled: should use undecorated storage",
			enableWatchCache:    false,
			watchCacheSizes:     []string{"crontabs.stable.example.com#0"},
			resource:            schema.GroupResource{Group: "stable.example.com", Resource: "crontabs"},
			expectStorageCacher: false,
			expectUndecorated:   true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			etcdOptions := genericoptions.EtcdOptions{
				StorageConfig: storagebackend.Config{
					Type:   "etcd3",
					Prefix: "/registry",
					Transport: storagebackend.TransportConfig{
						ServerList: []string{"http://127.0.0.1:2379"},
					},
					CompactionInterval:    storagebackend.DefaultCompactInterval,
					CountMetricPollPeriod: time.Minute,
				},
				DefaultStorageMediaType: "application/json",
				DeleteCollectionWorkers: 1,
				EnableGarbageCollection: true,
				EnableWatchCache:        tc.enableWatchCache,
				WatchCacheSizes:         tc.watchCacheSizes,
			}

			getter := NewCRDRESTOptionsGetter(etcdOptions, nil, nil)
			restOptions, err := getter.GetRESTOptions(tc.resource, nil)
			if err != nil {
				t.Fatalf("unexpected error from GetRESTOptions: %v", err)
			}

			isUndecorated := reflect.ValueOf(restOptions.Decorator).Pointer() == reflect.ValueOf(genericregistry.UndecoratedStorage).Pointer()
			if tc.expectUndecorated && !isUndecorated {
				t.Errorf("expected UndecoratedStorage decorator, got %v", restOptions.Decorator)
			}
			if tc.expectStorageCacher && isUndecorated {
				t.Errorf("expected decorated StorageWithCacher decorator, but got UndecoratedStorage (watch cache was unexpectedly disabled by WatchCacheSizes)")
			}
		})
	}
}

func TestNewCRDRESTOptionsGetterStorageObjectCountTracker(t *testing.T) {
	tracker := flowcontrolrequest.NewStorageObjectCountTracker()
	etcdOptions := genericoptions.EtcdOptions{
		StorageConfig: storagebackend.Config{
			Type:   "etcd3",
			Prefix: "/registry",
			Transport: storagebackend.TransportConfig{
				ServerList: []string{"http://127.0.0.1:2379"},
			},
			CompactionInterval:    storagebackend.DefaultCompactInterval,
			CountMetricPollPeriod: time.Minute,
		},
		DefaultStorageMediaType: "application/json",
		EnableWatchCache:        true,
	}

	getter := NewCRDRESTOptionsGetter(etcdOptions, nil, tracker)
	restOptions, err := getter.GetRESTOptions(schema.GroupResource{Group: "stable.example.com", Resource: "crontabs"}, nil)
	if err != nil {
		t.Fatalf("unexpected error from GetRESTOptions: %v", err)
	}

	if restOptions.StorageConfig.StorageObjectCountTracker != tracker {
		t.Errorf("expected StorageConfig.StorageObjectCountTracker to match provided tracker, got %v", restOptions.StorageConfig.StorageObjectCountTracker)
	}
	if restOptions.StorageObjectCountTracker != tracker {
		t.Errorf("expected restOptions.StorageObjectCountTracker to match provided tracker, got %v", restOptions.StorageObjectCountTracker)
	}
	if restOptions.StorageConfig.Codec == nil {
		t.Errorf("expected StorageConfig.Codec to be configured, got nil")
	}
}
