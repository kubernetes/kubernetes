/*
Copyright 2026 The Kubernetes Authors.

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
	"testing"

	genericoptions "k8s.io/apiserver/pkg/server/options"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

func TestNewCRDRESTOptionsGetter(t *testing.T) {
	etcdOptions := genericoptions.NewEtcdOptions(&storagebackend.Config{})
	etcdOptions.WatchCacheSizes = []string{"foos.example.com#0", "bars.example.com#100"}

	getter := NewCRDRESTOptionsGetter(*etcdOptions, nil, nil)
	factory, ok := getter.(*genericoptions.StorageFactoryRestOptionsFactory)
	if !ok {
		t.Fatalf("expected *genericoptions.StorageFactoryRestOptionsFactory, got %T", getter)
	}

	if factory.Options.WatchCacheSizes != nil {
		t.Errorf("expected WatchCacheSizes to be nil for custom resources, got %v", factory.Options.WatchCacheSizes)
	}
}
