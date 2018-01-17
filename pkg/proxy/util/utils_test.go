/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"net"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	api "k8s.io/kubernetes/pkg/apis/core"
	fake "k8s.io/kubernetes/pkg/proxy/util/testing"
)

func TestShouldSkipService(t *testing.T) {
	testCases := []struct {
		service    *api.Service
		svcName    types.NamespacedName
		shouldSkip bool
	}{
		{
			// Cluster IP is None
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: api.ClusterIPNone,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: true,
		},
		{
			// Cluster IP is empty
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "",
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: true,
		},
		{
			// ExternalName type service
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      api.ServiceTypeExternalName,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: true,
		},
		{
			// ClusterIP type service with ClusterIP set
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      api.ServiceTypeClusterIP,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: false,
		},
		{
			// NodePort type service with ClusterIP set
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      api.ServiceTypeNodePort,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: false,
		},
		{
			// LoadBalancer type service with ClusterIP set
			service: &api.Service{
				ObjectMeta: metav1.ObjectMeta{Namespace: "foo", Name: "bar"},
				Spec: api.ServiceSpec{
					ClusterIP: "1.2.3.4",
					Type:      api.ServiceTypeLoadBalancer,
				},
			},
			svcName:    types.NamespacedName{Namespace: "foo", Name: "bar"},
			shouldSkip: false,
		},
	}

	for i := range testCases {
		skip := ShouldSkipService(testCases[i].svcName, testCases[i].service)
		if skip != testCases[i].shouldSkip {
			t.Errorf("case %d: expect %v, got %v", i, testCases[i].shouldSkip, skip)
		}
	}
}

type InterfaceAddrsPair struct {
	itf   net.Interface
	addrs []net.Addr
}

func TestGetNodeAddressses(t *testing.T) {
	testCases := []struct {
		cidrs         []string
		nw            *fake.FakeNetwork
		itfAddrsPairs []InterfaceAddrsPair
		expected      sets.String
	}{
		{ // case 0
			cidrs: []string{"10.20.30.0/24"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "10.20.30.51/24"}},
				},
				{
					itf:   net.Interface{Index: 2, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "100.200.201.1/24"}},
				},
			},
			expected: sets.NewString("10.20.30.51"),
		},
		{ // case 1
			cidrs: []string{"0.0.0.0/0"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "10.20.30.51/24"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.0.1/8"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0"),
		},
		{ // case 2
			cidrs: []string{"2001:db8::/32", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "2001:db8::1/32"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("2001:db8::1", "::1"),
		},
		{ // case 3
			cidrs: []string{"::/0"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "2001:db8::1/32"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("::/0"),
		},
		{ // case 4
			cidrs: []string{"127.0.0.1/32"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "10.20.30.51/24"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.0.1/8"}},
				},
			},
			expected: sets.NewString("127.0.0.1"),
		},
		{ // case 5
			cidrs: []string{"127.0.0.0/8"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.1.1/8"}},
				},
			},
			expected: sets.NewString("127.0.1.1"),
		},
		{ // case 6
			cidrs: []string{"10.20.30.0/24", "100.200.201.0/24"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "10.20.30.51/24"}},
				},
				{
					itf:   net.Interface{Index: 2, MTU: 0, Name: "eth1", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "100.200.201.1/24"}},
				},
			},
			expected: sets.NewString("10.20.30.51", "100.200.201.1"),
		},
		{ // case 7
			cidrs: []string{"10.20.30.0/24", "100.200.201.0/24"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "192.168.1.2/24"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.0.1/8"}},
				},
			},
			expected: sets.NewString(),
		},
		{ // case 8
			cidrs: []string{},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "192.168.1.2/24"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "127.0.0.1/8"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0", "::/0"),
		},
		{ // case 9
			cidrs: []string{},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "2001:db8::1/32"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0", "::/0"),
		},
		{ // case 9
			cidrs: []string{"1.2.3.0/24", "0.0.0.0/0"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "1.2.3.4/30"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0"),
		},
		{ // case 10
			cidrs: []string{"0.0.0.0/0", "1.2.3.0/24", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "1.2.3.4/30"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("0.0.0.0/0", "::1"),
		},
		{ // case 11
			cidrs: []string{"::/0", "1.2.3.0/24", "::1/128"},
			nw:    fake.NewFakeNetwork(),
			itfAddrsPairs: []InterfaceAddrsPair{
				{
					itf:   net.Interface{Index: 0, MTU: 0, Name: "eth0", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "1.2.3.4/30"}},
				},
				{
					itf:   net.Interface{Index: 1, MTU: 0, Name: "lo", HardwareAddr: nil, Flags: 0},
					addrs: []net.Addr{fake.AddrStruct{Val: "::1/128"}},
				},
			},
			expected: sets.NewString("::/0", "1.2.3.4"),
		},
	}

	for i := range testCases {
		for _, pair := range testCases[i].itfAddrsPairs {
			testCases[i].nw.AddInterfaceAddr(&pair.itf, pair.addrs)
		}
		addrList, err := GetNodeAddresses(testCases[i].cidrs, testCases[i].nw)
		if err != nil {
			t.Errorf("case [%d], unexpected error: %v", i, err)
		}
		if !addrList.Equal(testCases[i].expected) {
			t.Errorf("case [%d], unexpected mismatch, expected: %v, got: %v", i, testCases[i].expected, addrList)
		}
	}
}
