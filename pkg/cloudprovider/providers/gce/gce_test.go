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

package gce

import (
	"reflect"
	"testing"

	compute "google.golang.org/api/compute/v1"
	"k8s.io/kubernetes/pkg/util/rand"
)

func TestGetRegion(t *testing.T) {
	zoneName := "us-central1-b"
	regionName, err := GetGCERegion(zoneName)
	if err != nil {
		t.Fatalf("unexpected error from GetGCERegion: %v", err)
	}
	if regionName != "us-central1" {
		t.Errorf("Unexpected region from GetGCERegion: %s", regionName)
	}
	gce := &GCECloud{
		localZone: zoneName,
		region:    regionName,
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
		testInstance := &gceInstance{
			Name: canonicalizeInstanceName(test.name),
			Zone: test.zone,
		}
		link2 := testInstance.makeComparableHostPath()
		if test.expectEqual && link1 != link2 {
			t.Errorf("expected link1 and link2 to be equal, got %s and %s", link1, link2)
		} else if !test.expectEqual && link1 == link2 {
			t.Errorf("expected link1 and link2 not to be equal, got %s and %s", link1, link2)
		}
	}
}

func TestScrubDNS(t *testing.T) {
	tcs := []struct {
		nameserversIn  []string
		searchesIn     []string
		nameserversOut []string
		searchesOut    []string
	}{
		{
			nameserversIn:  []string{"1.2.3.4", "5.6.7.8"},
			nameserversOut: []string{"1.2.3.4", "5.6.7.8"},
		},
		{
			searchesIn:  []string{"c.prj.internal.", "12345678910.google.internal.", "google.internal."},
			searchesOut: []string{"c.prj.internal.", "google.internal."},
		},
		{
			searchesIn:  []string{"c.prj.internal.", "12345678910.google.internal.", "zone.c.prj.internal.", "google.internal."},
			searchesOut: []string{"c.prj.internal.", "zone.c.prj.internal.", "google.internal."},
		},
		{
			searchesIn:  []string{"c.prj.internal.", "12345678910.google.internal.", "zone.c.prj.internal.", "google.internal.", "unexpected"},
			searchesOut: []string{"c.prj.internal.", "zone.c.prj.internal.", "google.internal.", "unexpected"},
		},
	}
	gce := &GCECloud{}
	for i := range tcs {
		n, s := gce.ScrubDNS(tcs[i].nameserversIn, tcs[i].searchesIn)
		if !reflect.DeepEqual(n, tcs[i].nameserversOut) {
			t.Errorf("Expected %v, got %v", tcs[i].nameserversOut, n)
		}
		if !reflect.DeepEqual(s, tcs[i].searchesOut) {
			t.Errorf("Expected %v, got %v", tcs[i].searchesOut, s)
		}
	}
}

func TestRestrictTargetPool(t *testing.T) {
	const maxInstances = 5
	tests := []struct {
		instances []string
		want      []string
	}{
		{
			instances: []string{"1", "2", "3", "4", "5"},
			want:      []string{"1", "2", "3", "4", "5"},
		},
		{
			instances: []string{"1", "2", "3", "4", "5", "6"},
			want:      []string{"4", "3", "5", "2", "6"},
		},
	}
	for _, tc := range tests {
		rand.Seed(5)
		got := restrictTargetPool(append([]string{}, tc.instances...), maxInstances)
		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("restrictTargetPool(%v) => %v, want %v", tc.instances, got, tc.want)
		}
	}
}

func TestComputeUpdate(t *testing.T) {
	const maxInstances = 5
	const fakeZone = "us-moon1-f"
	tests := []struct {
		tp           []string
		instances    []string
		wantToAdd    []string
		wantToRemove []string
	}{
		{
			// Test adding all instances.
			tp:           []string{},
			instances:    []string{"0", "1", "2"},
			wantToAdd:    []string{"0", "1", "2"},
			wantToRemove: []string{},
		},
		{
			// Test node 1 coming back healthy.
			tp:           []string{"0", "2"},
			instances:    []string{"0", "1", "2"},
			wantToAdd:    []string{"1"},
			wantToRemove: []string{},
		},
		{
			// Test node 1 going healthy while node 4 needs to be removed.
			tp:           []string{"0", "2", "4"},
			instances:    []string{"0", "1", "2"},
			wantToAdd:    []string{"1"},
			wantToRemove: []string{"4"},
		},
		{
			// Test exceeding the TargetPool max of 5 (for the test),
			// which shuffles in 7, 5, 8 based on the deterministic
			// seed below.
			tp:           []string{"0", "2", "4", "6"},
			instances:    []string{"0", "1", "2", "3", "5", "7", "8"},
			wantToAdd:    []string{"7", "5", "8"},
			wantToRemove: []string{"4", "6"},
		},
		{
			// Test all nodes getting removed.
			tp:           []string{"0", "1", "2", "3"},
			instances:    []string{},
			wantToAdd:    []string{},
			wantToRemove: []string{"0", "1", "2", "3"},
		},
	}
	for _, tc := range tests {
		rand.Seed(5) // Arbitrary RNG seed for deterministic testing.

		// Dummy up the gceInstance slice.
		var instances []*gceInstance
		for _, inst := range tc.instances {
			instances = append(instances, &gceInstance{Name: inst, Zone: fakeZone})
		}
		// Dummy up the TargetPool URL list.
		var urls []string
		for _, inst := range tc.tp {
			inst := &gceInstance{Name: inst, Zone: fakeZone}
			urls = append(urls, inst.makeComparableHostPath())
		}
		gotAddInsts, gotRem := computeUpdate(&compute.TargetPool{Instances: urls}, instances, maxInstances)
		var wantAdd []string
		for _, inst := range tc.wantToAdd {
			inst := &gceInstance{Name: inst, Zone: fakeZone}
			wantAdd = append(wantAdd, inst.makeComparableHostPath())
		}
		var gotAdd []string
		for _, inst := range gotAddInsts {
			gotAdd = append(gotAdd, inst.Instance)
		}
		if !reflect.DeepEqual(wantAdd, gotAdd) {
			t.Errorf("computeTargetPool(%v, %v) => added %v, wanted %v", tc.tp, tc.instances, gotAdd, wantAdd)
		}
		_ = gotRem
		// var gotRem []string
		// for _, inst := range gotRemInsts {
		// 	gotRem = append(gotRem, inst.Instance)
		// }
		// if !reflect.DeepEqual(tc.wantToRemove, gotRem) {
		// 	t.Errorf("computeTargetPool(%v, %v) => removed %v, wanted %v", tc.tp, tc.instances, gotRem, tc.wantToRemove)
		// }
	}
}
