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
	"testing"
)

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

func TestGetHostTag(t *testing.T) {
	tests := []struct {
		host     string
		expected string
	}{
		{
			host:     "kubernetes-minion-559o",
			expected: "kubernetes-minion",
		},
		{
			host:     "gke-test-ea6e8c80-node-8ytk",
			expected: "gke-test-ea6e8c80-node",
		},
		{
			host:     "kubernetes-minion-559o.c.PROJECT_NAME.internal",
			expected: "kubernetes-minion",
		},
	}

	gce := &GCECloud{}
	for _, test := range tests {
		hostTag := gce.computeHostTag(test.host)
		if hostTag != test.expected {
			t.Errorf("expected: %s, saw: %s for %s", test.expected, hostTag, test.host)
		}
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
