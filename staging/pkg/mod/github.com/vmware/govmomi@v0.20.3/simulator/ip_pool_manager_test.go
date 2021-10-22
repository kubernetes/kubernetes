/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"context"
	"fmt"
	"net"
	"reflect"
	"testing"

	"github.com/vmware/govmomi"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/types"
)

func TestIpPoolv4(t *testing.T) {
	tests := []struct {
		ipRange string
		length  int32
	}{
		{"10.10.10.2#250", 250},
		{"10.10.10.2#10, 10.10.10.20#20", 30},
		{"10.10.10.2#10, 10.10.10.20#20, 10.10.10.50#20", 50},
	}

	for _, test := range tests {
		var ipPool = MustNewIpPool(&types.IpPool{
			Id:                     1,
			Name:                   "ip-pool",
			AvailableIpv4Addresses: test.length,
			AvailableIpv6Addresses: 0,
			AllocatedIpv6Addresses: 0,
			AllocatedIpv4Addresses: 0,
			Ipv4Config: &types.IpPoolIpPoolConfigInfo{
				Netmask:       "10.10.10.255",
				Gateway:       "10.10.10.1",
				SubnetAddress: "10.10.10.0",
				Range:         test.ipRange,
			},
		})

		if len(ipPool.ipv4Pool) != int(test.length) {
			t.Fatalf("expect length to be %d; got %d", test.length, len(ipPool.ipv4Pool))
		}

		ip, err := ipPool.AllocateIPv4("alloc")
		if err != nil {
			t.Fatal(err)
		}

		ip2, err := ipPool.AllocateIPv4("alloc")
		if err != nil {
			t.Fatal(err)
		}
		if ip != ip2 {
			t.Fatalf("same allocation key should allocate the same ip; got %s, %s", ip, ip2)
		}

		err = ipPool.ReleaseIpv4("bad-alloc")
		if err == nil {
			t.Fatal("expect error to release a bad allocation")
		}

		if len(ipPool.ipv4Pool) != int(test.length)-1 {
			t.Fatalf("expect length to be %d; got %d", test.length-1, len(ipPool.ipv4Pool))
		}

		err = ipPool.ReleaseIpv4("alloc")
		if err != nil {
			t.Fatal(err)
		}

		if len(ipPool.ipv4Pool) != int(test.length) {
			t.Fatalf("expect length to be %d; got %d", test.length, len(ipPool.ipv4Pool))
		}

		allocated := map[string]bool{}
		for i := 0; i < int(test.length); i++ {
			ip, err := ipPool.AllocateIPv4(fmt.Sprintf("alloc-%d", i))
			if err != nil {
				t.Fatal(err)
			}

			if _, ok := allocated[ip]; ok {
				t.Fatalf("duplicated allocation of ip %q", ip)
			}
			allocated[ip] = true
		}

		_, err = ipPool.AllocateIPv4("last-allocation")
		if err != errNoIpAvailable {
			t.Fatalf("expect errNoIpAvailable; got %s", err)
		}
	}
}

func TestIpPoolv6(t *testing.T) {
	tests := []struct {
		ipRange string
		length  int32
	}{
		{"2001:4860:0:2001::2#250", 250},
	}

	for _, test := range tests {
		var ipPool = MustNewIpPool(&types.IpPool{
			Id:                     1,
			Name:                   "ip-pool",
			AvailableIpv4Addresses: 0,
			AvailableIpv6Addresses: test.length,
			AllocatedIpv6Addresses: 0,
			AllocatedIpv4Addresses: 0,
			Ipv6Config: &types.IpPoolIpPoolConfigInfo{
				Netmask:       "2001:4860:0:2001::ff",
				Gateway:       "2001:4860:0:2001::1",
				SubnetAddress: "2001:4860:0:2001::0",
				Range:         test.ipRange,
			},
		})

		if len(ipPool.ipv6Pool) != int(test.length) {
			t.Fatalf("expect length to be %d; got %d", test.length, len(ipPool.ipv4Pool))
		}

		ip, err := ipPool.AllocateIpv6("alloc")
		if err != nil {
			t.Fatal(err)
		}

		ip2, err := ipPool.AllocateIpv6("alloc")
		if err != nil {
			t.Fatal(err)
		}
		if ip != ip2 {
			t.Fatalf("same allocation key should allocate the same ip; got %s, %s", ip, ip2)
		}

		err = ipPool.ReleaseIpv6("bad-alloc")
		if err == nil {
			t.Fatal("expect error to release a bad allocation")
		}

		if len(ipPool.ipv6Pool) != int(test.length)-1 {
			t.Fatalf("expect length to be %d; got %d", test.length-1, len(ipPool.ipv4Pool))
		}

		err = ipPool.ReleaseIpv6("alloc")
		if err != nil {
			t.Fatal(err)
		}

		if len(ipPool.ipv6Pool) != int(test.length) {
			t.Fatalf("expect length to be %d; got %d", test.length, len(ipPool.ipv4Pool))
		}

		allocated := map[string]bool{}
		for i := 0; i < int(test.length); i++ {
			ip, err := ipPool.AllocateIpv6(fmt.Sprintf("alloc-%d", i))
			if err != nil {
				t.Fatal(err)
			}

			if _, ok := allocated[ip]; ok {
				t.Fatalf("duplicated allocation of ip %q", ip)
			}
			allocated[ip] = true
		}

		_, err = ipPool.AllocateIpv6("last-allocation")
		if err != errNoIpAvailable {
			t.Fatalf("expect errNoIpAvailable; got %s", err)
		}
	}
}

func TestIpPoolManagerLifecycle(t *testing.T) {
	ctx := context.Background()
	m := VPX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	ref := types.ManagedObjectReference{Type: "IpPoolManager", Value: "IpPoolManager"}

	var ipPool = &types.IpPool{
		Name:                   "ip-pool",
		AvailableIpv4Addresses: 250,
		AvailableIpv6Addresses: 250,
		AllocatedIpv4Addresses: 0,
		AllocatedIpv6Addresses: 0,
		Ipv4Config: &types.IpPoolIpPoolConfigInfo{
			Netmask:       "10.10.10.255",
			Gateway:       "10.10.10.1",
			SubnetAddress: "10.10.10.0",
			Range:         "10.10.10.2#250",
		},
		Ipv6Config: &types.IpPoolIpPoolConfigInfo{
			Netmask:       "2001:4860:0:2001::ff",
			Gateway:       "2001:4860:0:2001::1",
			SubnetAddress: "2001:4860:0:2001::0",
			Range:         "2001:4860:0:2001::2#250",
		},
	}

	createReq := &types.CreateIpPool{
		This: ref,
		Pool: *ipPool,
	}

	createResp, err := methods.CreateIpPool(ctx, c.Client, createReq)
	if err != nil {
		t.Fatal(err)
	}
	if createResp.Returnval != 2 {
		t.Fatalf("expect pool id to be 2; got %d", createResp.Returnval)
	}

	ipPool.Id = 2
	ipPool.Ipv4Config = &types.IpPoolIpPoolConfigInfo{
		Netmask:       "10.20.10.255",
		Gateway:       "10.20.10.1",
		SubnetAddress: "10.20.10.0",
		Range:         "10.20.10.2#250",
	}

	updateReq := &types.UpdateIpPool{
		This: ref,
		Pool: *ipPool,
	}

	_, err = methods.UpdateIpPool(ctx, c.Client, updateReq)
	if err != nil {
		t.Fatal(err)
	}

	queryReq := &types.QueryIpPools{
		This: ref,
	}

	queryResp, err := methods.QueryIpPools(ctx, c.Client, queryReq)
	if err != nil {
		t.Fatal(err)
	}

	if len(queryResp.Returnval) != 2 {
		t.Fatalf("expect length of ip pools is 2; got %d", len(queryResp.Returnval))
	}
	if !reflect.DeepEqual(queryResp.Returnval[1].Ipv4Config, ipPool.Ipv4Config) {
		t.Fatalf("expect query result equal to %+v; got %+v",
			ipPool.Ipv4Config, queryResp.Returnval[1].Ipv4Config)
	}

	destroyReq := &types.DestroyIpPool{
		This: ref,
		Id:   2,
	}

	_, err = methods.DestroyIpPool(ctx, c.Client, destroyReq)
	if err != nil {
		t.Fatal(err)
	}

	queryResp, err = methods.QueryIpPools(ctx, c.Client, queryReq)
	if err != nil {
		t.Fatal(err)
	}
	if len(queryResp.Returnval) != 1 {
		t.Fatalf("expect length of ip pools is 1 (1 deleted); got %d", len(queryResp.Returnval))
	}
}

func TestIpPoolManagerAllocate(t *testing.T) {
	ctx := context.Background()
	m := VPX()

	defer m.Remove()

	err := m.Create()
	if err != nil {
		t.Fatal(err)
	}

	s := m.Service.NewServer()
	defer s.Close()

	c, err := govmomi.NewClient(ctx, s.URL, true)
	if err != nil {
		t.Fatal(err)
	}

	ref := types.ManagedObjectReference{Type: "IpPoolManager", Value: "IpPoolManager"}

	// Allocate IPv4
	allocateReq := &types.AllocateIpv4Address{
		This:         ref,
		PoolId:       1,
		AllocationId: "alloc",
	}

	allocateResp, err := methods.AllocateIpv4Address(ctx, c.Client, allocateReq)
	if err != nil {
		t.Fatal(err)
	}
	if net.ParseIP(allocateResp.Returnval) == nil {
		t.Fatalf("%q is not IP address", allocateResp.Returnval)
	}

	releaseReq := &types.ReleaseIpAllocation{
		This:         ref,
		PoolId:       1,
		AllocationId: "alloc",
	}

	queryReq := &types.QueryIPAllocations{
		This:         ref,
		PoolId:       1,
		ExtensionKey: "alloc",
	}

	queryResp, err := methods.QueryIPAllocations(ctx, c.Client, queryReq)
	if err != nil {
		t.Fatal(err)
	}
	if len(queryResp.Returnval) != 1 {
		t.Fatalf("expect length of result 1; got %s", queryResp.Returnval)
	}
	if queryResp.Returnval[0].IpAddress != allocateResp.Returnval {
		t.Fatalf("expect same IP address; got %s, %s", queryResp.Returnval[0].IpAddress, allocateResp.Returnval)
	}

	_, err = methods.ReleaseIpAllocation(ctx, c.Client, releaseReq)
	if err != nil {
		t.Fatal(err)
	}
}
