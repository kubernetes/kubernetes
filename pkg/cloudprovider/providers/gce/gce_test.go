/*
Copyright 2014 The Kubernetes Authors.

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

func TestSplitProviderID(t *testing.T) {
	providers := []struct {
		providerID string

		project  string
		zone     string
		instance string

		fail bool
	}{
		{
			providerID: ProviderName + "://project-example-164317/us-central1-f/kubernetes-node-fhx1",
			project:    "project-example-164317",
			zone:       "us-central1-f",
			instance:   "kubernetes-node-fhx1",
			fail:       false,
		},
		{
			providerID: ProviderName + "://project-example.164317/us-central1-f/kubernetes-node-fhx1",
			project:    "project-example.164317",
			zone:       "us-central1-f",
			instance:   "kubernetes-node-fhx1",
			fail:       false,
		},
		{
			providerID: ProviderName + "://project-example-164317/us-central1-fkubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: ProviderName + ":/project-example-164317/us-central1-f/kubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: "aws://project-example-164317/us-central1-f/kubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: ProviderName + "://project-example-164317/us-central1-f/kubernetes-node-fhx1/",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: ProviderName + "://project-example.164317//kubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
		{
			providerID: ProviderName + "://project-example.164317/kubernetes-node-fhx1",
			project:    "",
			zone:       "",
			instance:   "",
			fail:       true,
		},
	}

	for _, test := range providers {
		project, zone, instance, err := splitProviderID(test.providerID)
		if (err != nil) != test.fail {
			t.Errorf("Expected to failt=%t, with pattern %v", test.fail, test)
		}

		if test.fail {
			continue
		}

		if project != test.project {
			t.Errorf("Expected %v, but got %v", test.project, project)
		}
		if zone != test.zone {
			t.Errorf("Expected %v, but got %v", test.zone, zone)
		}
		if instance != test.instance {
			t.Errorf("Expected %v, but got %v", test.instance, instance)
		}
	}
}
