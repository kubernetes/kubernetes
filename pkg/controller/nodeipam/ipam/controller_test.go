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

package ipam

import (
	"net"
	"testing"

	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/cidrset"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam/test"
)

func TestOccupyServiceCIDR(t *testing.T) {
	const clusterCIDR = "10.1.0.0/16"

TestCase:
	for _, tc := range []struct {
		serviceCIDR string
	}{
		{"10.0.255.0/24"},
		{"10.1.0.0/24"},
		{"10.1.255.0/24"},
		{"10.2.0.0/24"},
	} {
		serviceCIDR := test.MustParseCIDR(tc.serviceCIDR)
		set, err := cidrset.NewCIDRSet(test.MustParseCIDR(clusterCIDR), 24)
		if err != nil {
			t.Errorf("test case %+v: NewCIDRSet() = %v, want nil", tc, err)
		}
		if err := occupyServiceCIDR(set, test.MustParseCIDR(clusterCIDR), serviceCIDR); err != nil {
			t.Errorf("test case %+v: occupyServiceCIDR() = %v, want nil", tc, err)
		}
		// Allocate until full.
		var cidrs []*net.IPNet
		for {
			cidr, err := set.AllocateNext()
			if err != nil {
				if err == cidrset.ErrCIDRRangeNoCIDRsRemaining {
					break
				}
				t.Errorf("set.AllocateNext() = %v, want %v", err, cidrset.ErrCIDRRangeNoCIDRsRemaining)
				continue TestCase
			}
			cidrs = append(cidrs, cidr)
		}
		// No allocated CIDR range should intersect with serviceCIDR.
		for _, c := range cidrs {
			if c.Contains(serviceCIDR.IP) || serviceCIDR.Contains(c.IP) {
				t.Errorf("test case %+v: allocated CIDR %v from service range", tc, c)
			}
		}
	}
}
