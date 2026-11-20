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

package endpointslice

import (
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	discovery "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/endpointslice/topologycache"
	"k8s.io/klog/v2/ktesting"
	"k8s.io/utils/ptr"
)

func pendingAllocation(addressType discovery.AddressType) *topologycache.SliceInfo {
	si := &topologycache.SliceInfo{ServiceKey: "ns/svc", AddressType: addressType, ToCreate: []*discovery.EndpointSlice{{AddressType: addressType}}}
	for i, zone := range []string{"zone-a", "zone-b"} {
		address := fmt.Sprintf("10.0.0.%d", i+1)
		if addressType == discovery.AddressTypeIPv6 {
			address = fmt.Sprintf("2001:db8::%x", i+1)
		}
		si.ToCreate[0].Endpoints = append(si.ToCreate[0].Endpoints, discovery.Endpoint{
			Addresses: []string{address}, Zone: ptr.To(zone), Conditions: discovery.EndpointConditions{Ready: ptr.To(true)},
		})
	}
	return si
}

func TestReconcilerForgetsPendingAllocations(t *testing.T) {
	for _, deleted := range []bool{false, true} {
		name := "removed address type without an existing slice"
		if deleted {
			name = "deleted Service"
		}
		t.Run(name, func(t *testing.T) {
			logger, _ := ktesting.NewTestContext(t)
			client := fake.NewSimpleClientset()
			client.PrependReactor("create", "endpointslices", func(action clienttesting.Action) (bool, runtime.Object, error) {
				slice := action.(clienttesting.CreateAction).GetObject().(*discovery.EndpointSlice)
				slice.Name = "svc-test"
				return true, slice, nil
			})
			r := newReconciler(client, nil, 100)
			r.topologyCache = topologycache.NewTopologyCache()
			r.topologyCache.AddHints(logger, pendingAllocation(discovery.AddressTypeIPv6))
			if deleted {
				r.topologyCache.AddHints(logger, pendingAllocation(discovery.AddressTypeIPv4))
				r.DeleteService("ns", "svc")
			} else {
				service := &v1.Service{
					ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "svc", UID: types.UID("svc"), Annotations: map[string]string{v1.AnnotationTopologyMode: "Auto"}},
					Spec:       v1.ServiceSpec{IPFamilies: []v1.IPFamily{v1.IPv4Protocol}, Ports: []v1.ServicePort{{Port: 80, TargetPort: intstr.FromInt32(80)}}},
				}
				if err := r.Reconcile(logger, service, nil, nil, time.Time{}); err != nil {
					t.Fatal(err)
				}
			}
			var nodes []*v1.Node
			for i, zone := range []string{"zone-a", "zone-b"} {
				nodes = append(nodes, &v1.Node{
					ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("node-%d", i), Labels: map[string]string{v1.LabelTopologyZone: zone}},
					Status:     v1.NodeStatus{Allocatable: v1.ResourceList{v1.ResourceCPU: resource.MustParse("4")}, Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}},
				})
			}
			r.topologyCache.SetNodes(logger, nodes)
			if got := r.topologyCache.GetOverloadedServices(); len(got) != 0 {
				t.Fatalf("discarded allocations still request reconciliation: %v", got)
			}
		})
	}
}
