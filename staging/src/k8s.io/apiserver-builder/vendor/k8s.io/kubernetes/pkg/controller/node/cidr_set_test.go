/*
Copyright 2016 The Kubernetes Authors.

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

package node

import (
	"math/big"
	"net"
	"reflect"
	"testing"

	"github.com/golang/glog"
)

func TestCIDRSetFullyAllocated(t *testing.T) {
	_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/30")
	a := newCIDRSet(clusterCIDR, 30)

	p, err := a.allocateNext()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.String() != "127.123.234.0/30" {
		t.Fatalf("unexpected allocated cidr: %s", p.String())
	}

	_, err = a.allocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}

	a.release(p)
	p, err = a.allocateNext()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if p.String() != "127.123.234.0/30" {
		t.Fatalf("unexpected allocated cidr: %s", p.String())
	}
	_, err = a.allocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}
}

func TestCIDRSet_RandomishAllocation(t *testing.T) {
	_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/16")
	a := newCIDRSet(clusterCIDR, 24)
	// allocate all the CIDRs
	var err error
	cidrs := make([]*net.IPNet, 256)

	for i := 0; i < 256; i++ {
		cidrs[i], err = a.allocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	_, err = a.allocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}
	// release them all
	for i := 0; i < 256; i++ {
		a.release(cidrs[i])
	}

	// allocate the CIDRs again
	rcidrs := make([]*net.IPNet, 256)
	for i := 0; i < 256; i++ {
		rcidrs[i], err = a.allocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %d, %v", i, err)
		}
	}
	_, err = a.allocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}

	if !reflect.DeepEqual(cidrs, rcidrs) {
		t.Fatalf("expected re-allocated cidrs are the same collection")
	}
}

func TestCIDRSet_AllocationOccupied(t *testing.T) {
	_, clusterCIDR, _ := net.ParseCIDR("127.123.234.0/16")
	a := newCIDRSet(clusterCIDR, 24)

	// allocate all the CIDRs
	var err error
	cidrs := make([]*net.IPNet, 256)

	for i := 0; i < 256; i++ {
		cidrs[i], err = a.allocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	_, err = a.allocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}
	// release them all
	for i := 0; i < 256; i++ {
		a.release(cidrs[i])
	}
	// occupy the last 128 CIDRs
	for i := 128; i < 256; i++ {
		a.occupy(cidrs[i])
	}

	// allocate the first 128 CIDRs again
	rcidrs := make([]*net.IPNet, 128)
	for i := 0; i < 128; i++ {
		rcidrs[i], err = a.allocateNext()
		if err != nil {
			t.Fatalf("unexpected error: %d, %v", i, err)
		}
	}
	_, err = a.allocateNext()
	if err == nil {
		t.Fatalf("expected error because of fully-allocated range")
	}

	// check Occupy() work properly
	for i := 128; i < 256; i++ {
		rcidrs = append(rcidrs, cidrs[i])
	}
	if !reflect.DeepEqual(cidrs, rcidrs) {
		t.Fatalf("expected re-allocated cidrs are the same collection")
	}
}

func TestGetBitforCIDR(t *testing.T) {
	cases := []struct {
		clusterCIDRStr string
		subNetMaskSize int
		subNetCIDRStr  string
		expectedBit    int
		expectErr      bool
	}{
		{
			clusterCIDRStr: "127.0.0.0/8",
			subNetMaskSize: 16,
			subNetCIDRStr:  "127.0.0.0/16",
			expectedBit:    0,
			expectErr:      false,
		},
		{
			clusterCIDRStr: "127.0.0.0/8",
			subNetMaskSize: 16,
			subNetCIDRStr:  "127.123.0.0/16",
			expectedBit:    123,
			expectErr:      false,
		},
		{
			clusterCIDRStr: "127.0.0.0/8",
			subNetMaskSize: 16,
			subNetCIDRStr:  "127.168.0.0/16",
			expectedBit:    168,
			expectErr:      false,
		},
		{
			clusterCIDRStr: "127.0.0.0/8",
			subNetMaskSize: 16,
			subNetCIDRStr:  "127.224.0.0/16",
			expectedBit:    224,
			expectErr:      false,
		},
		{
			clusterCIDRStr: "192.168.0.0/16",
			subNetMaskSize: 24,
			subNetCIDRStr:  "192.168.12.0/24",
			expectedBit:    12,
			expectErr:      false,
		},
		{
			clusterCIDRStr: "192.168.0.0/16",
			subNetMaskSize: 24,
			subNetCIDRStr:  "192.168.151.0/24",
			expectedBit:    151,
			expectErr:      false,
		},
		{
			clusterCIDRStr: "192.168.0.0/16",
			subNetMaskSize: 24,
			subNetCIDRStr:  "127.168.224.0/24",
			expectErr:      true,
		},
	}

	for _, tc := range cases {
		_, clusterCIDR, err := net.ParseCIDR(tc.clusterCIDRStr)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		cs := newCIDRSet(clusterCIDR, tc.subNetMaskSize)

		_, subnetCIDR, err := net.ParseCIDR(tc.subNetCIDRStr)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		got, err := cs.getIndexForCIDR(subnetCIDR)
		if err == nil && tc.expectErr {
			glog.Errorf("expected error but got null")
			continue
		}

		if err != nil && !tc.expectErr {
			glog.Errorf("unexpected error: %v", err)
			continue
		}

		if got != tc.expectedBit {
			glog.Errorf("expected %v, but got %v", tc.expectedBit, got)
		}
	}
}

func TestOccupy(t *testing.T) {
	cases := []struct {
		clusterCIDRStr    string
		subNetMaskSize    int
		subNetCIDRStr     string
		expectedUsedBegin int
		expectedUsedEnd   int
		expectErr         bool
	}{
		{
			clusterCIDRStr:    "127.0.0.0/8",
			subNetMaskSize:    16,
			subNetCIDRStr:     "127.0.0.0/8",
			expectedUsedBegin: 0,
			expectedUsedEnd:   256,
			expectErr:         false,
		},
		{
			clusterCIDRStr:    "127.0.0.0/8",
			subNetMaskSize:    16,
			subNetCIDRStr:     "127.0.0.0/2",
			expectedUsedBegin: 0,
			expectedUsedEnd:   256,
			expectErr:         false,
		},
		{
			clusterCIDRStr:    "127.0.0.0/8",
			subNetMaskSize:    16,
			subNetCIDRStr:     "127.0.0.0/16",
			expectedUsedBegin: 0,
			expectedUsedEnd:   0,
			expectErr:         false,
		},
		{
			clusterCIDRStr:    "127.0.0.0/8",
			subNetMaskSize:    32,
			subNetCIDRStr:     "127.0.0.0/16",
			expectedUsedBegin: 0,
			expectedUsedEnd:   65535,
			expectErr:         false,
		},
		{
			clusterCIDRStr:    "127.0.0.0/7",
			subNetMaskSize:    16,
			subNetCIDRStr:     "127.0.0.0/15",
			expectedUsedBegin: 256,
			expectedUsedEnd:   257,
			expectErr:         false,
		},
		{
			clusterCIDRStr:    "127.0.0.0/7",
			subNetMaskSize:    15,
			subNetCIDRStr:     "127.0.0.0/15",
			expectedUsedBegin: 128,
			expectedUsedEnd:   128,
			expectErr:         false,
		},
		{
			clusterCIDRStr:    "127.0.0.0/7",
			subNetMaskSize:    18,
			subNetCIDRStr:     "127.0.0.0/15",
			expectedUsedBegin: 1024,
			expectedUsedEnd:   1031,
			expectErr:         false,
		},
	}

	for _, tc := range cases {
		_, clusterCIDR, err := net.ParseCIDR(tc.clusterCIDRStr)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		cs := newCIDRSet(clusterCIDR, tc.subNetMaskSize)

		_, subnetCIDR, err := net.ParseCIDR(tc.subNetCIDRStr)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		err = cs.occupy(subnetCIDR)
		if err == nil && tc.expectErr {
			t.Errorf("expected error but got none")
			continue
		}
		if err != nil && !tc.expectErr {
			t.Errorf("unexpected error: %v", err)
			continue
		}

		expectedUsed := big.Int{}
		for i := tc.expectedUsedBegin; i <= tc.expectedUsedEnd; i++ {
			expectedUsed.SetBit(&expectedUsed, i, 1)
		}
		if expectedUsed.Cmp(&cs.used) != 0 {
			t.Errorf("error")
		}
	}
}
