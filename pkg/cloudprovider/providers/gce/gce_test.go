/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package gce_cloud

import (
	"net"
	"testing"

	compute "google.golang.org/api/compute/v1"
)

func TestOwnsAddress(t *testing.T) {
	tests := []struct {
		ip        net.IP
		addrs     []*compute.Address
		expectOwn bool
	}{
		{
			ip:        net.ParseIP("1.2.3.4"),
			addrs:     []*compute.Address{},
			expectOwn: false,
		},
		{
			ip: net.ParseIP("1.2.3.4"),
			addrs: []*compute.Address{
				{Address: "2.3.4.5"},
				{Address: "2.3.4.6"},
				{Address: "2.3.4.7"},
			},
			expectOwn: false,
		},
		{
			ip: net.ParseIP("2.3.4.5"),
			addrs: []*compute.Address{
				{Address: "2.3.4.5"},
				{Address: "2.3.4.6"},
				{Address: "2.3.4.7"},
			},
			expectOwn: true,
		},
		{
			ip: net.ParseIP("2.3.4.6"),
			addrs: []*compute.Address{
				{Address: "2.3.4.5"},
				{Address: "2.3.4.6"},
				{Address: "2.3.4.7"},
			},
			expectOwn: true,
		},
		{
			ip: net.ParseIP("2.3.4.7"),
			addrs: []*compute.Address{
				{Address: "2.3.4.5"},
				{Address: "2.3.4.6"},
				{Address: "2.3.4.7"},
			},
			expectOwn: true,
		},
	}
	for _, test := range tests {
		own := ownsAddress(test.ip, test.addrs)
		if own != test.expectOwn {
			t.Errorf("expected: %v, got %v for %v", test.expectOwn, own, test)
		}
	}
}

func TestGetRegion(t *testing.T) {
	gce := &GCECloud{
		zone: "us-central1-b",
	}
	zones, ok := gce.Zones()
	if !ok {
		t.Fatalf("Unexpected missing zones impl")
	}
	zone, err := zones.GetZone()
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if zone.Region != "us-central1" {
		t.Errorf("Unexpected region: %s", zone.Region)
	}
}

func TestComparingHostURLs(t *testing.T) {
	tests := []struct {
		host1       string
		zone        string
		name        string
		expectEqual bool
	}{
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: true,
		},
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/cool-project/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: true,
		},
		{
			host1:       "https://www.googleapis.com/compute/v23/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: true,
		},
		{
			host1:       "https://www.googleapis.com/compute/v24/projects/1234567/regions/us-central1/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: true,
		},
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-c",
			name:        "kubernetes-node-fhx1",
			expectEqual: false,
		},
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx1",
			expectEqual: false,
		},
		{
			host1:       "https://www.googleapis.com/compute/v1/projects/1234567/zones/us-central1-f/instances/kubernetes-node-fhx1",
			zone:        "us-central1-f",
			name:        "kubernetes-node-fhx",
			expectEqual: false,
		},
	}

	for _, test := range tests {
		link1 := hostURLToComparablePath(test.host1)
		link2 := makeComparableHostPath(test.zone, test.name)
		if test.expectEqual && link1 != link2 {
			t.Errorf("expected link1 and link2 to be equal, got %s and %s", link1, link2)
		} else if !test.expectEqual && link1 == link2 {
			t.Errorf("expected link1 and link2 not to be equal, got %s and %s", link1, link2)
		}
	}
}
