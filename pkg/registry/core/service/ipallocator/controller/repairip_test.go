/*
Copyright 2023 The Kubernetes Authors.

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

package controller

import (
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	testclock "k8s.io/utils/clock/testing"
)

var (
	serviceCIDRv4 = "10.0.0.0/16"
	serviceCIDRv6 = "2001:db8::/64"

	// testTimeNow is used to lock the fake clocks view of the current time.
	testTimeNow = time.Date(2012, 1, 1, 0, 0, 0, 0, time.UTC)
)

type fakeRepair struct {
	*RepairIPAddress
	serviceStore     cache.Store
	ipAddressStore   cache.Store
	serviceCIDRStore cache.Store
}

func newFakeRepair(testTime time.Time) (*fake.Clientset, *fakeRepair) {
	fakeClient := fake.NewSimpleClientset()

	informerFactory := informers.NewSharedInformerFactory(fakeClient, 0*time.Second)
	serviceInformer := informerFactory.Core().V1().Services()
	serviceIndexer := serviceInformer.Informer().GetIndexer()

	serviceCIDRInformer := informerFactory.Networking().V1().ServiceCIDRs()
	serviceCIDRIndexer := serviceCIDRInformer.Informer().GetIndexer()

	ipInformer := informerFactory.Networking().V1().IPAddresses()
	ipIndexer := ipInformer.Informer().GetIndexer()

	fakeClient.PrependReactor("create", "ipaddresses", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		ip := action.(k8stesting.CreateAction).GetObject().(*networkingv1.IPAddress)
		err := ipIndexer.Add(ip)
		return false, ip, err
	}))
	fakeClient.PrependReactor("update", "ipaddresses", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		ip := action.(k8stesting.UpdateAction).GetObject().(*networkingv1.IPAddress)
		return false, ip, fmt.Errorf("IPAddress is inmutable after creation")
	}))
	fakeClient.PrependReactor("delete", "ipaddresses", k8stesting.ReactionFunc(func(action k8stesting.Action) (bool, runtime.Object, error) {
		ip := action.(k8stesting.DeleteAction).GetName()
		err := ipIndexer.Delete(ip)
		return false, &networkingv1.IPAddress{}, err
	}))
	fakeClock := testclock.NewFakeClock(testTime)
	r := newRepairIPAddress(0*time.Second,
		fakeClient,
		serviceInformer,
		serviceCIDRInformer,
		ipInformer,
		fakeClock,
	)
	return fakeClient, &fakeRepair{r, serviceIndexer, ipIndexer, serviceCIDRIndexer}
}

func TestRepairServiceIP(t *testing.T) {
	tests := []struct {
		name        string
		svcs        []*v1.Service
		ipAddresses []*networkingv1.IPAddress
		cidrs       []*networkingv1.ServiceCIDR
		testTime    time.Time
		expectedIPs []string
		actions     [][]string // verb and resource
		events      []string
	}{
		{
			name: "no changes needed single stack",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc", []string{"10.0.1.1"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"10.0.1.1"},
			actions:     [][]string{},
			events:      []string{},
		},
		{
			name: "no changes needed dual stack",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1", "2001:db8::10"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc", []string{"10.0.1.1"}), testTimeNow),
				newIPAddressWithCreationTimestamp("2001:db8::10", newService("test-svc", []string{"2001:db8::10"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"10.0.1.1", "2001:db8::10"},
			actions:     [][]string{},
			events:      []string{},
		},
		{
			name: "no changes needed dual stack multiple cidrs",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"192.168.0.1", "2001:db8:a:b::10"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("192.168.0.1", newService("test-svc", []string{"192.168.0.1"}), testTimeNow),
				newIPAddressWithCreationTimestamp("2001:db8:a:b::10", newService("test-svc", []string{"2001:db8:a:b::10"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
				newServiceCIDR("custom", "192.168.0.0/24", "2001:db8:a:b::/64"),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"192.168.0.1", "2001:db8:a:b::10"},
			actions:     [][]string{},
			events:      []string{},
		},
		// these two cases simulate migrating from bitmaps to IPAddress objects
		{
			name: "create IPAddress single stack",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1"}, testTimeNow.Add(1*time.Second)),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"10.0.1.1"},
			actions:     [][]string{{"create", "ipaddresses"}},
			events:      []string{"Warning ClusterIPNotAllocated Cluster IP [IPv4]: 10.0.1.1 is not allocated; repairing"},
		},
		{
			name: "create IPAddresses dual stack",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1", "2001:db8::10"}, testTimeNow.Add(1*time.Second)),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"10.0.1.1", "2001:db8::10"},
			actions:     [][]string{{"create", "ipaddresses"}, {"create", "ipaddresses"}},
			events: []string{
				"Warning ClusterIPNotAllocated Cluster IP [IPv4]: 10.0.1.1 is not allocated; repairing",
				"Warning ClusterIPNotAllocated Cluster IP [IPv6]: 2001:db8::10 is not allocated; repairing",
			},
		},
		{
			name: "create IPAddress single stack from secondary",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"192.168.1.1"}, testTimeNow.Add(1*time.Second)),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
				newServiceCIDR("custom", "192.168.1.0/24", ""),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"192.168.1.1"},
			actions:     [][]string{{"create", "ipaddresses"}},
			events:      []string{"Warning ClusterIPNotAllocated Cluster IP [IPv4]: 192.168.1.1 is not allocated; repairing"},
		},
		{
			name: "reconcile IPAddress single stack wrong reference",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc2", []string{"10.0.1.1"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"10.0.1.1"},
			actions:     [][]string{{"delete", "ipaddresses"}, {"create", "ipaddresses"}},
			events:      []string{"Warning ClusterIPNotAllocated the ClusterIP [IPv4]: 10.0.1.1 for Service bar/test-svc has a wrong reference; repairing"},
		},
		{
			name: "reconcile IPAddresses dual stack",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1", "2001:db8::10"}, testTimeNow.Add(-1*time.Second))},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc2", []string{"10.0.1.1"}), testTimeNow),
				newIPAddressWithCreationTimestamp("2001:db8::10", newService("test-svc2", []string{"2001:db8::10"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"10.0.1.1", "2001:db8::10"},
			actions:     [][]string{{"delete", "ipaddresses"}, {"create", "ipaddresses"}, {"delete", "ipaddresses"}, {"create", "ipaddresses"}},
			events: []string{
				"Warning ClusterIPNotAllocated the ClusterIP [IPv4]: 10.0.1.1 for Service bar/test-svc has a wrong reference; repairing",
				"Warning ClusterIPNotAllocated the ClusterIP [IPv6]: 2001:db8::10 for Service bar/test-svc has a wrong reference; repairing",
			},
		},
		{
			name: "one IP out of range",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"192.168.1.1", "2001:db8::10"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("192.168.1.1", newService("test-svc", []string{"192.168.1.1"}), testTimeNow),
				newIPAddressWithCreationTimestamp("2001:db8::10", newService("test-svc", []string{"2001:db8::10"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"2001:db8::10"},
			actions:     [][]string{},
			events:      []string{"Warning ClusterIPOutOfRange Cluster IP [IPv4]: 192.168.1.1 is not within any configured Service CIDR; please recreate service"},
		},
		{
			name: "one IP orphan within grace period",
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc", []string{"10.0.1.1"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
		},
		{
			name: "one IP orphan after grace period",
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc", []string{"10.0.1.1"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			// we update the fake clock here simulating the IPAddress was created more than 70 seconds ago,
			// this simulates grace period that prevents ip address cleanup is over.
			testTime: testTimeNow.Add(70 * time.Second),
			actions:  [][]string{{"delete", "ipaddresses"}},
			events:   []string{"Warning IPAddressNotAllocated IPAddress: 10.0.1.1 for Service bar/test-svc appears to have leaked: cleaning up"},
		},
		{
			name: "one IP out of range matching the network address",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.0.0"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.0.0", newService("test-svc", []string{"10.0.0.0"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"10.0.0.0"},
			actions:     [][]string{},
			events:      []string{"Warning ClusterIPOutOfRange Cluster IP [IPv4]: 10.0.0.0 is not within any configured Service CIDR; please recreate service"},
		},
		{
			name: "one IP out of range matching the broadcast address",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.255.255"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.255.255", newService("test-svc", []string{"10.0.255.255"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"10.0.255.255"},
			actions:     [][]string{},
			events:      []string{"Warning ClusterIPOutOfRange Cluster IP [IPv4]: 10.0.255.255 is not within any configured Service CIDR; please recreate service"},
		},
		{
			name: "one IPv6 out of range matching the subnet address",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"2001:db8::"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("2001:db8::", newService("test-svc", []string{"2001:db8::"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"2001:db8::"},
			actions:     [][]string{},
			events:      []string{"Warning ClusterIPOutOfRange Cluster IP [IPv6]: 2001:db8:: is not within any configured Service CIDR; please recreate service"},
		},
		{
			name: "one IPv6 matching the broadcast address",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"2001:db8::ffff:ffff:ffff:ffff"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("2001:db8::ffff:ffff:ffff:ffff", newService("test-svc", []string{"2001:db8::ffff:ffff:ffff:ffff"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime:    testTimeNow,
			expectedIPs: []string{"2001:db8::ffff:ffff:ffff:ffff"},
		},
		{
			name: "one IP orphan matching the broadcast address within grace period",
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.255.255", newService("test-svc", []string{"10.0.255.255"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime: testTimeNow,
		},
		{
			name: "one IP orphan matching the broadcast address after grace period",
			ipAddresses: []*networkingv1.IPAddress{
				// we update the fake clock here simulating the IPAddress was created more than 90 seconds ago,
				// this simulates grace period that prevents ip address cleanup is over.
				newIPAddressWithCreationTimestamp("10.0.255.255", newService("test-svc", []string{"10.0.255.255"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime: testTimeNow.Add(90 * time.Second),
			actions:  [][]string{{"delete", "ipaddresses"}},
			events:   []string{"Warning IPAddressNotAllocated IPAddress: 10.0.255.255 for Service bar/test-svc appears to have leaked: cleaning up"},
		},
		{
			name: "Two IPAddresses referencing the same service",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc", []string{"10.0.1.1"}), testTimeNow),
				newIPAddressWithCreationTimestamp("10.0.1.2", newService("test-svc", []string{"10.0.1.1"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime: testTimeNow,
			actions:  [][]string{{"delete", "ipaddresses"}},
			events:   []string{"Warning IPAddressWrongReference IPAddress: 10.0.1.2 for Service bar/test-svc has a wrong reference; cleaning up"},
		},
		{
			name: "Two IPAddresses referencing the same service when service create is called again and ip address is in grace period",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc", []string{"10.0.1.1"}), testTimeNow),
				// ensure the second IPAddress object has a creation timestamp greater than service creation timestamp
				newIPAddressWithCreationTimestamp("10.0.1.2", newService("test-svc", []string{"10.0.1.1"}), testTimeNow.Add(2*time.Second)),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			// update the test time, ensuring 5 seconds have not passed after creation of the second IPAddress Object
			testTime: testTimeNow.Add(3 * time.Second),
		},
		{
			name: "Two IPAddresses referencing the same service when service create is called again and ip address is not in grace period",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc", []string{"10.0.1.1"}), testTimeNow),
				// ensure the second IPAddress object has a creation timestamp greater than service creation timestamp
				newIPAddressWithCreationTimestamp("10.0.1.2", newService("test-svc", []string{"10.0.1.1"}), testTimeNow.Add(2*time.Second)),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			// update the test time, ensuring 5 seconds have passed after creation of the second IPAddress Object
			testTime: testTimeNow.Add(10 * time.Second),
			actions:  [][]string{{"delete", "ipaddresses"}},
			events:   []string{"Warning IPAddressWrongReference IPAddress: 10.0.1.2 for Service bar/test-svc has a wrong reference; cleaning up"},
		},
		{
			name: "Two Services with same ClusterIP",
			svcs: []*v1.Service{
				newServiceWithCreationTimestamp("test-svc", []string{"10.0.1.1"}, testTimeNow.Add(1*time.Second)),
				newServiceWithCreationTimestamp("test-svc2", []string{"10.0.1.1"}, testTimeNow.Add(1*time.Second)),
			},
			ipAddresses: []*networkingv1.IPAddress{
				newIPAddressWithCreationTimestamp("10.0.1.1", newService("test-svc2", []string{"10.0.1.1"}), testTimeNow),
			},
			cidrs: []*networkingv1.ServiceCIDR{
				newServiceCIDR("kubernetes", serviceCIDRv4, serviceCIDRv6),
			},
			testTime: testTimeNow,
			events:   []string{"Warning ClusterIPAlreadyAllocated Cluster IP [IPv4]:10.0.1.1 was assigned to multiple services; please recreate service"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {

			c, r := newFakeRepair(test.testTime)
			// add cidrs
			for _, cidr := range test.cidrs {
				err := r.serviceCIDRStore.Add(cidr)
				if err != nil {
					t.Errorf("Unexpected error trying to add Service %v object: %v", cidr, err)
				}
			}

			// override for testing
			r.servicesSynced = func() bool { return true }
			r.ipAddressSynced = func() bool { return true }
			r.serviceCIDRSynced = func() bool { return true }
			recorder := events.NewFakeRecorder(100)
			r.recorder = recorder
			for _, svc := range test.svcs {
				err := r.serviceStore.Add(svc)
				if err != nil {
					t.Errorf("Unexpected error trying to add Service %v object: %v", svc, err)
				}
			}

			for _, ip := range test.ipAddresses {
				err := r.ipAddressStore.Add(ip)
				if err != nil {
					t.Errorf("Unexpected error trying to add IPAddress %s object: %v", ip, err)
				}
			}

			if err := r.runOnce(); err != nil {
				t.Fatal(err)
			}

			for _, ip := range test.expectedIPs {
				_, err := r.ipAddressLister.Get(ip)
				if err != nil {
					t.Errorf("Unexpected error trying to get IPAddress %s object: %v", ip, err)
				}
			}

			expectAction(t, c.Actions(), test.actions)
			expectEvents(t, recorder.Events, test.events)
		})
	}

}

func TestRepairIPAddress_syncIPAddress(t *testing.T) {
	tests := []struct {
		name     string
		ip       *networkingv1.IPAddress
		testTime time.Time
		actions  [][]string // verb and resource
		wantErr  bool
	}{
		{
			name: "correct ipv4 address",
			ip: &networkingv1.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "10.0.1.1",
					Labels: map[string]string{
						networkingv1.LabelIPAddressFamily: string(v1.IPv4Protocol),
						networkingv1.LabelManagedBy:       ipallocator.ControllerName,
					},
					CreationTimestamp: metav1.Time{Time: testTimeNow},
				},
				Spec: networkingv1.IPAddressSpec{
					ParentRef: &networkingv1.ParentReference{
						Group:     "",
						Resource:  "services",
						Name:      "foo",
						Namespace: "bar",
					},
				},
			},
			testTime: testTimeNow,
		},
		{
			name: "correct ipv6 address",
			ip: &networkingv1.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "2001:db8::11",
					Labels: map[string]string{
						networkingv1.LabelIPAddressFamily: string(v1.IPv6Protocol),
						networkingv1.LabelManagedBy:       ipallocator.ControllerName,
					},
					CreationTimestamp: metav1.Time{Time: testTimeNow},
				},
				Spec: networkingv1.IPAddressSpec{
					ParentRef: &networkingv1.ParentReference{
						Group:     "",
						Resource:  "services",
						Name:      "foo",
						Namespace: "bar",
					},
				},
			},
			testTime: testTimeNow,
		},
		{
			name: "not managed by this controller",
			ip: &networkingv1.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "2001:db8::11",
					Labels: map[string]string{
						networkingv1.LabelIPAddressFamily: string(v1.IPv6Protocol),
						networkingv1.LabelManagedBy:       "controller-foo",
					},
					CreationTimestamp: metav1.Time{Time: testTimeNow},
				},
				Spec: networkingv1.IPAddressSpec{
					ParentRef: &networkingv1.ParentReference{
						Group:     "networking.gateway.k8s.io",
						Resource:  "gateway",
						Name:      "foo",
						Namespace: "bar",
					},
				},
			},
			testTime: testTimeNow,
		},
		{
			name: "out of range",
			ip: &networkingv1.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "fd00:db8::11",
					Labels: map[string]string{
						networkingv1.LabelIPAddressFamily: string(v1.IPv6Protocol),
						networkingv1.LabelManagedBy:       ipallocator.ControllerName,
					},
					CreationTimestamp: metav1.Time{Time: testTimeNow},
				},
				Spec: networkingv1.IPAddressSpec{
					ParentRef: &networkingv1.ParentReference{
						Group:     "",
						Resource:  "services",
						Name:      "foo",
						Namespace: "bar",
					},
				},
			},
			testTime: testTimeNow,
		},
		{
			name: "leaked ip",
			ip: &networkingv1.IPAddress{
				ObjectMeta: metav1.ObjectMeta{
					Name: "10.0.1.1",
					Labels: map[string]string{
						networkingv1.LabelIPAddressFamily: string(v1.IPv6Protocol),
						networkingv1.LabelManagedBy:       ipallocator.ControllerName,
					},
					CreationTimestamp: metav1.Time{Time: testTimeNow},
				},
				Spec: networkingv1.IPAddressSpec{
					ParentRef: &networkingv1.ParentReference{
						Group:     "",
						Resource:  "services",
						Name:      "noexist",
						Namespace: "bar",
					},
				},
			},
			// we update the fake clock here simulating the IPAddress was created more than 90 seconds ago,
			// this simulates grace period that prevents ip address cleanup is over.
			testTime: testTimeNow.Add(90 * time.Second),
			actions:  [][]string{{"delete", "ipaddresses"}},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c, r := newFakeRepair(tt.testTime)
			err := r.ipAddressStore.Add(tt.ip)
			if err != nil {
				t.Fatal(err)
			}
			err = r.serviceStore.Add(newService("foo", []string{tt.ip.Name}))
			if err != nil {
				t.Fatal(err)
			}

			// override for testing
			r.servicesSynced = func() bool { return true }
			r.ipAddressSynced = func() bool { return true }
			recorder := events.NewFakeRecorder(100)
			r.recorder = recorder
			if err := r.syncIPAddress(tt.ip.Name); (err != nil) != tt.wantErr {
				t.Errorf("RepairIPAddress.syncIPAddress() error = %v, wantErr %v", err, tt.wantErr)
			}
			expectAction(t, c.Actions(), tt.actions)

		})
	}
}

func newIPAddressWithCreationTimestamp(name string, svc *v1.Service, timestamp time.Time) *networkingv1.IPAddress {
	ipAddress := newIPAddress(name, svc)
	ipAddress.CreationTimestamp = metav1.Time{Time: timestamp}
	return ipAddress
}

func newServiceWithCreationTimestamp(name string, ips []string, timestamp time.Time) *v1.Service {
	service := newService(name, ips)
	service.CreationTimestamp = metav1.Time{Time: timestamp}
	return service
}

func newService(name string, ips []string) *v1.Service {
	if len(ips) == 0 {
		return nil
	}
	svc := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:         "bar",
			Name:              name,
			CreationTimestamp: metav1.Time{Time: time.Now()}},
		Spec: v1.ServiceSpec{
			ClusterIP:  ips[0],
			ClusterIPs: ips,
			Type:       v1.ServiceTypeClusterIP,
		},
	}
	return svc
}

func newServiceCIDR(name, primary, secondary string) *networkingv1.ServiceCIDR {
	serviceCIDR := &networkingv1.ServiceCIDR{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: networkingv1.ServiceCIDRSpec{},
	}
	serviceCIDR.Spec.CIDRs = append(serviceCIDR.Spec.CIDRs, primary)
	if secondary != "" {
		serviceCIDR.Spec.CIDRs = append(serviceCIDR.Spec.CIDRs, secondary)
	}
	return serviceCIDR
}

func expectAction(t *testing.T, actions []k8stesting.Action, expected [][]string) {
	t.Helper()
	if len(actions) != len(expected) {
		t.Fatalf("Expected at least %d actions, got %d \ndiff: %v", len(expected), len(actions), cmp.Diff(expected, actions))
	}

	for i, action := range actions {
		verb := expected[i][0]
		if action.GetVerb() != verb {
			t.Errorf("Expected action %d verb to be %s, got %s", i, verb, action.GetVerb())
		}
		resource := expected[i][1]
		if action.GetResource().Resource != resource {
			t.Errorf("Expected action %d resource to be %s, got %s", i, resource, action.GetResource().Resource)
		}
	}
}

func expectEvents(t *testing.T, actual <-chan string, expected []string) {
	t.Helper()
	c := time.After(wait.ForeverTestTimeout)
	for _, e := range expected {
		select {
		case a := <-actual:
			if e != a {
				t.Errorf("Expected event %q, got %q", e, a)
				return
			}
		case <-c:
			t.Errorf("Expected event %q, got nothing", e)
			// continue iterating to print all expected events
		}
	}
	for {
		select {
		case a := <-actual:
			t.Errorf("Unexpected event: %q", a)
		default:
			return // No more events, as expected.
		}
	}
}
