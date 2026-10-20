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
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/dynamic-resource-allocation/cel"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

type mutableSliceLister struct {
	mu     sync.Mutex
	slices []*resourceapi.ResourceSlice
}

func (l *mutableSliceLister) ListPatchedResourceSlices() ([]*resourceapi.ResourceSlice, error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	return append([]*resourceapi.ResourceSlice(nil), l.slices...), nil
}

func (l *mutableSliceLister) ListWithDeviceTaintRules() ([]*resourceapi.ResourceSlice, error) {
	return l.ListPatchedResourceSlices()
}

func (l *mutableSliceLister) set(slices ...*resourceapi.ResourceSlice) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.slices = slices
}

func publishRaceSlice(pool string, generation int64, expression string) *resourceapi.ResourceSlice {
	return &resourceapi.ResourceSlice{
		ObjectMeta: metav1.ObjectMeta{Name: pool},
		Spec: resourceapi.ResourceSliceSpec{
			Driver: "driver.example.com",
			Pool:   resourceapi.ResourcePool{Name: pool, Generation: generation, ResourceSliceCount: 1},
			SharingAffinity: []resourceapi.SharingAffinityExtractor{{
				Name: "subnet", APIGroup: "config.example.com", Kind: "Network", Expression: expression,
			}},
		},
	}
}

func newPublishRaceTracker(t *testing.T, lister *mutableSliceLister) *claimTracker {
	logger, _ := ktesting.NewTestContext(t)
	c := &claimTracker{logger: logger, metadataSlices: lister}
	c.metadata = &affinityMetadata{pools: make(map[poolName]poolMeta)}
	c.allocatedDevices = newAllocatedDevices(logger, nil)
	c.allocatedDevices.metadata = c.metadata
	return c
}

func publishRaceCommitted(c *claimTracker) metaSnap {
	c.allocatedDevices.mutex.RLock()
	defer c.allocatedDevices.mutex.RUnlock()
	pools := make(map[poolName]poolMeta, len(c.metadata.pools))
	for k, v := range c.metadata.pools {
		pools[k] = v
	}
	return metaSnap{synced: c.metadata.synced, pools: pools}
}

type publishRaceManager struct {
	claims  fwk.ResourceClaimTracker
	slices  fwk.ResourceSliceLister
	classes fwk.DeviceClassLister
}

func (m publishRaceManager) ResourceClaims() fwk.ResourceClaimTracker { return m.claims }
func (m publishRaceManager) ResourceSlices() fwk.ResourceSliceLister  { return m.slices }
func (m publishRaceManager) DeviceClasses() fwk.DeviceClassLister     { return m.classes }
func (m publishRaceManager) DeviceClassResolver() fwk.DeviceClassResolver {
	return nil
}

// Init lists old metadata, then a handler fires for a newer slice. The handler
// must wait for init to commit and then publish the newest view; init must not
// overwrite it afterwards.
func TestAffinityMetadataInitRaceWithHandler(t *testing.T) {
	oldSlices := poolEventsSlices()
	newSlices := []*resourceapi.ResourceSlice{oldSlices[0].DeepCopy(), oldSlices[1].DeepCopy()}
	for _, slice := range newSlices {
		slice.Spec.Pool.Generation++
	}
	lister := &mutableSliceLister{slices: oldSlices}
	c, _ := newPoolEventsUnitTracker(t, oldSlices...)
	c.metadataSlices = lister
	c.affinity.slices = lister

	paused := make(chan struct{})
	resume := make(chan struct{})
	c.publishAfterListHook = func(source string) {
		if source == "init" {
			close(paused)
			<-resume
		}
	}
	initDone := make(chan struct{})
	go func() {
		defer close(initDone)
		c.publishMetadata(true, nil)
	}()
	<-paused

	lister.set(newSlices...)
	handlerDone := make(chan struct{})
	go func() {
		defer close(handlerDone)
		c.publishMetadata(false, affectedPools(oldSlices[0], oldSlices[1], newSlices[0], newSlices[1]))
	}()
	select {
	case <-handlerDone:
		t.Fatal("handler publication completed while init held the publisher lock")
	case <-time.After(50 * time.Millisecond):
	}
	close(resume)
	<-initDone
	<-handlerDone

	committed := publishRaceCommitted(c)
	if !committed.synced {
		t.Fatal("registry not synced after init")
	}
	key := poolName{accountingDriver, accountingPool}
	want := metadataView(newSlices, key)
	if got := committed.pools[key]; !sameMetadata(got, want) {
		t.Fatalf("committed metadata = %+v, want newest view %+v", got, want)
	}
	if !committed.coherent(newSlices) {
		t.Fatal("registry not coherent with lister without another slice event")
	}

	claim := reserveClaim("init-race", "blue")
	claim.Status.Allocation = nil
	assumecache.AddTestObject(c.cache, claim)
	pod := reservePod("init-race-pod", claim)
	node := framework.NewNodeInfo()
	node.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: workerNode.Name}})
	pl := &DynamicResources{
		draManager: publishRaceManager{
			claims:  c,
			slices:  lister,
			classes: filterAffinityClasses{class: &resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}}},
		},
		fts:      feature.Features{EnableDRASharingAffinity: true, EnableDRAConsumableCapacity: true},
		celCache: cel.NewCache(10, cel.Features{}),
	}
	state := framework.NewCycleState()
	_, ctx := ktesting.NewTestContext(t)
	if _, status := pl.PreFilter(ctx, state, pod, []fwk.NodeInfo{node}); !status.IsSuccess() {
		t.Fatalf("PreFilter after init race = %v, want success against new view without another event", status)
	}
	if status := pl.Filter(ctx, state, pod, node); !status.IsSuccess() {
		t.Fatalf("Filter after init race = %v, want success against new view without another event", status)
	}
	if status := pl.Reserve(ctx, state, pod, workerNode.Name); !status.IsSuccess() {
		t.Fatalf("Reserve after init race = %v, want success against new view without another event", status)
	}
	if got := c.GetPendingAllocation(claim.UID); got == nil {
		t.Fatalf("Reserve after init race did not publish pending allocation: %#v", got)
	}
}

// Concurrent event publications with interleaved lister updates must leave the
// registry equal to the final lister view.
func TestAffinityPublishSerializedNoStaleOverwrite(t *testing.T) {
	const pools, rounds = 4, 50
	lister := &mutableSliceLister{}
	c := newPublishRaceTracker(t, lister)
	c.publishMetadata(true, nil)

	var wg sync.WaitGroup
	for p := range pools {
		wg.Add(1)
		go func() {
			defer wg.Done()
			name := fmt.Sprintf("p%d", p)
			for gen := int64(1); gen <= rounds; gen++ {
				slice := publishRaceSlice(name, gen, fmt.Sprintf("%q", fmt.Sprint(gen)))
				lister.mu.Lock()
				next := make([]*resourceapi.ResourceSlice, 0, len(lister.slices)+1)
				for _, s := range lister.slices {
					if s.Spec.Pool.Name != name {
						next = append(next, s)
					}
				}
				lister.slices = append(next, slice)
				lister.mu.Unlock()
				c.publishMetadata(false, affectedPools(slice))
			}
		}()
	}
	wg.Wait()

	final, _ := lister.ListPatchedResourceSlices()
	committed := publishRaceCommitted(c)
	if !committed.coherent(final) {
		t.Fatalf("registry %+v diverged from final lister view", committed.pools)
	}
	for p := range pools {
		if got := committed.pools[poolName{"driver.example.com", fmt.Sprintf("p%d", p)}].generation; got != rounds {
			t.Errorf("pool p%d generation = %d, want %d", p, got, rounds)
		}
	}
}
