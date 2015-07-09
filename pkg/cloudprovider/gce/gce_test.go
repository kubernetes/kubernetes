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
	}

	gce := &GCECloud{}
	for _, test := range tests {
		hostTag := gce.computeHostTag(test.host)
		if hostTag != test.expected {
			t.Errorf("expected: %s, saw: %s for %s", test.expected, hostTag, test.host)
		}
	}
}
