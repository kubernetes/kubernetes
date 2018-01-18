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

package gce

import (
	"net"
	"reflect"
	"testing"

	compute "google.golang.org/api/compute/v1"
)

func TestLastIPInRange(t *testing.T) {
	for _, tc := range []struct {
		cidr string
		want string
	}{
		{"10.1.2.3/32", "10.1.2.3"},
		{"10.1.2.0/31", "10.1.2.1"},
		{"10.1.0.0/30", "10.1.0.3"},
		{"10.0.0.0/29", "10.0.0.7"},
		{"::0/128", "::"},
		{"::0/127", "::1"},
		{"::0/126", "::3"},
		{"::0/120", "::ff"},
	} {
		_, c, err := net.ParseCIDR(tc.cidr)
		if err != nil {
			t.Errorf("net.ParseCIDR(%v) = _, %v, %v; want nil", tc.cidr, c, err)
			continue
		}

		if lastIP := lastIPInRange(c); lastIP.String() != tc.want {
			t.Errorf("LastIPInRange(%v) = %v; want %v", tc.cidr, lastIP, tc.want)
		}
	}
}

func TestSubnetsInCIDR(t *testing.T) {
	subnets := []*compute.Subnetwork{
		{
			Name:        "A",
			IpCidrRange: "10.0.0.0/20",
		},
		{
			Name:        "B",
			IpCidrRange: "10.0.16.0/20",
		},
		{
			Name:        "C",
			IpCidrRange: "10.132.0.0/20",
		},
		{
			Name:        "D",
			IpCidrRange: "10.0.32.0/20",
		},
		{
			Name:        "E",
			IpCidrRange: "10.134.0.0/20",
		},
	}
	expectedNames := []string{"C", "E"}

	gotSubs, err := subnetsInCIDR(subnets, autoSubnetIPRange)
	if err != nil {
		t.Errorf("autoSubnetInList() = _, %v", err)
	}

	var gotNames []string
	for _, v := range gotSubs {
		gotNames = append(gotNames, v.Name)
	}
	if !reflect.DeepEqual(gotNames, expectedNames) {
		t.Errorf("autoSubnetInList() = %v, expected: %v", gotNames, expectedNames)
	}
}
