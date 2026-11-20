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

package topologycache

import (
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func recoveryNodes() []*v1.Node {
	var nodes []*v1.Node
	for i, zone := range []string{"zone-a", "zone-b"} {
		nodes = append(nodes, &v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("node-%d", i), Labels: map[string]string{v1.LabelTopologyZone: zone}},
			Status: v1.NodeStatus{
				Allocatable: v1.ResourceList{v1.ResourceCPU: resource.MustParse("4")},
				Conditions:  []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
			},
		})
	}
	return nodes
}

func recoverySliceInfo(addressType discovery.AddressType) *SliceInfo {
	si := &SliceInfo{ServiceKey: "ns/svc", AddressType: addressType, ToCreate: []*discovery.EndpointSlice{{AddressType: addressType}}}
	for i := 0; i < 6; i++ {
		address := fmt.Sprintf("10.0.0.%d", i+1)
		if addressType == discovery.AddressTypeIPv6 {
			address = fmt.Sprintf("2001:db8::%x", i+1)
		}
		si.ToCreate[0].Endpoints = append(si.ToCreate[0].Endpoints, discovery.Endpoint{
			Addresses: []string{address}, Zone: ptr.To(fmt.Sprintf("zone-%c", 'a'+i/3)),
			Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
		})
	}
	return si
}

func expectRetryServices(t *testing.T, cache *TopologyCache, want []string) {
	t.Helper()
	if got := cache.GetOverloadedServices(); !cmp.Equal(got, want) {
		t.Fatalf("Services needing reconciliation = %v, want %v", got, want)
	}
}

func TestUnallocatedAddressTypesRecoverIndependently(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	cache := NewTopologyCache()
	for _, addressType := range []discovery.AddressType{discovery.AddressTypeIPv4, discovery.AddressTypeIPv6} {
		cache.AddHints(logger, recoverySliceInfo(addressType))
	}
	if cache.HasPopulatedHints("ns/svc") {
		t.Fatal("failed allocations were reported as populated hints")
	}
	expectRetryServices(t, cache, []string{})
	cache.SetNodes(logger, recoveryNodes())
	expectRetryServices(t, cache, []string{"ns/svc"})
	_, _, events := cache.AddHints(logger, recoverySliceInfo(discovery.AddressTypeIPv4))
	if len(events) != 1 || events[0].Reason != "TopologyAwareHintsEnabled" {
		t.Fatalf("recovering hints did not emit the enabled event: %+v", events)
	}
	expectRetryServices(t, cache, []string{"ns/svc"})
	cache.RemoveHints("ns/svc", discovery.AddressTypeIPv6)
	if !cache.HasPopulatedHints("ns/svc") {
		t.Fatal("removing a pending address type hid populated hints for another address type")
	}
	expectRetryServices(t, cache, []string{})
	_, _, events = cache.AddHints(logger, recoverySliceInfo(discovery.AddressTypeIPv4))
	if len(events) != 0 {
		t.Fatalf("unchanged populated hints emitted an enabled event again: %+v", events)
	}
	cache.RemoveHints("ns/svc", discovery.AddressTypeIPv4)
	if cache.HasPopulatedHints("ns/svc") {
		t.Fatal("last address type remained populated after removal")
	}
	expectRetryServices(t, cache, []string{})
}

func TestTopologyChangeDuringAllocationIsNotLost(t *testing.T) {
	for _, recoverDuringAllocation := range []bool{false, true} {
		name := "topology still cannot allocate hints"
		if recoverDuringAllocation {
			name = "topology recovers before failed allocation is recorded"
		}
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			cache := NewTopologyCache()
			started, release, done := make(chan struct{}), make(chan struct{}), make(chan struct{})
			var startedOnce, releaseOnce sync.Once
			unblock := func() { releaseOnce.Do(func() { close(release) }) }
			t.Cleanup(func() {
				unblock()
				select {
				case <-done:
				case <-time.After(5 * time.Second):
					t.Error("allocation goroutine did not stop")
				}
			})
			blockingLogger := funcr.New(func(_, _ string) {
				startedOnce.Do(func() { close(started); <-release })
			}, funcr.Options{})
			go func() {
				cache.AddHints(blockingLogger, recoverySliceInfo(discovery.AddressTypeIPv4))
				close(done)
			}()
			select {
			case <-started:
			case <-time.After(5 * time.Second):
				t.Fatal("allocation did not reach the barrier")
			}
			if recoverDuringAllocation {
				cache.SetNodes(logger, recoveryNodes())
			} else {
				cache.SetNodes(logger, nil)
			}
			expectRetryServices(t, cache, []string{"ns/svc"})
			unblock()
			select {
			case <-done:
			case <-time.After(5 * time.Second):
				t.Fatal("allocation did not finish")
			}
			if !recoverDuringAllocation {
				expectRetryServices(t, cache, []string{})
				cache.SetNodes(logger, recoveryNodes())
			}
			expectRetryServices(t, cache, []string{"ns/svc"})
			cache.AddHints(logger, recoverySliceInfo(discovery.AddressTypeIPv4))
			expectRetryServices(t, cache, []string{})
		})
	}
}
