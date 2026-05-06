/*
Copyright 2025 The Kubernetes Authors.

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

package volumebinding

import (
	"testing"

	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/testutils/ktesting"
)

func TestPVAssumeCache(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	informer := &testInformer{
		indexer: cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{}),
		t:       t,
	}
	cache, err := NewPVAssumeCache(logger, informer)
	if err != nil {
		t.Fatalf("Failed to create PV cache: %v", err)
	}
	informer.add(makePV("pv1", "sc1").withVersion("1").PersistentVolume)

	verifyPVs := func(cache PVAssumeCache) {
		pvs, err := cache.ListPVs("sc1")
		if err != nil {
			t.Fatalf("Failed to list PVs: %v", err)
		}
		if len(pvs) != 1 || pvs[0].Name != "pv1" {
			t.Errorf("Unexpected PVs: %v", pvs)
		}
	}
	verifyPVs(cache)

	// Shared informer
	cache2, err := NewPVAssumeCache(logger, informer)
	if err != nil {
		t.Fatalf("Failed to create PV cache2: %v", err)
	}
	verifyPVs(cache2)
}
