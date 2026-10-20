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

package dynamicresources

import (
	"fmt"
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
)

type staticSliceLister []*resourceapi.ResourceSlice

func (s staticSliceLister) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	return s, nil
}

// Ordinary events must only recompute the affected pool; init recomputes all.
func TestPublishMetadataScaling(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	const pools = 2000
	var slices staticSliceLister
	for i := range pools {
		slices = append(slices, &resourceapi.ResourceSlice{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("s%d", i)},
			Spec: resourceapi.ResourceSliceSpec{
				Driver: "driver.example.com",
				Pool:   resourceapi.ResourcePool{Name: fmt.Sprintf("p%d", i), Generation: 1, ResourceSliceCount: 1},
			},
		})
	}
	c := &claimTracker{logger: logger, metadataSlices: slices}
	c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
	c.allocatedDevices = newAllocatedDevices(logger, nil)
	c.allocatedDevices.metadata = c.metadata
	computed := 0
	c.publishBeforeCommitHook = func(string, string) { computed++ }

	for _, slice := range slices {
		c.publishMetadata(false, affectedPools(slice))
	}
	if computed != pools {
		t.Fatalf("event publication computed %d pool views, want %d", computed, pools)
	}
	computed = 0
	c.publishMetadata(true, nil)
	if computed != pools {
		t.Fatalf("init publication computed %d pool views, want %d", computed, pools)
	}
	if !c.metadata.synced || len(c.metadata.pools) != pools {
		t.Fatalf("synced=%v pools=%d", c.metadata.synced, len(c.metadata.pools))
	}
	snap, _ := func() (metaSnap, error) {
		c.allocatedDevices.mutex.RLock()
		defer c.allocatedDevices.mutex.RUnlock()
		return metaSnap{synced: true, pools: c.metadata.pools}, nil
	}()
	if !snap.coherent(slices) {
		t.Fatal("published metadata not coherent with slices")
	}
}
