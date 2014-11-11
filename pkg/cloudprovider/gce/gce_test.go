/*
Copyright 2014 Google Inc. All rights reserved.

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
	compute "code.google.com/p/google-api-go-client/compute/v1"
	"fmt"
	"net/http"
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

func TestGetZone(t *testing.T) {
	gce := &GCECloud{
		zone: "us-central1-b",
	}
	_, err := gce.GetZone()

	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
}

func TestgetGceRegion(t *testing.T) {
	zone := "us-central1-b"
	region, err := getGceRegion(zone)

	if err != nil {
		t.Fatalf("error")
	}

	if region != "central1-b" {
		t.Fatalf("Expected %s, got %s", "central1-b", region)
	}
}

type MockGCECloud struct {
	gceCloud *GCECloud
}

func (mockgce *MockGCECloud) makeTargetPool(name, region string, hosts []string) (string, error) {
	link := fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/regions/%s/targetPools/%s",
		mockgce.gceCloud.projectID,
		region, name)
	return link, nil
}

func (mockgce *MockGCECloud) CreateTCPLoadBalancer(name, region string, port int, hosts []string) error {
	return mockgce.gceCloud.CreateTCPLoadBalancer(name, region, port, hosts)
}

func TestCreateTCPLoadBalancer(t *testing.T) {
	var mockClient = &http.Client{
		Transport: http.DefaultTransport,
	}

	svc, _ := compute.New(mockClient)
	gce := &GCECloud{
		zone:      "us-central1-b",
		service:   svc,
		projectID: "testID",
	}
	host := []string{"host1"}
	mockgce := MockGCECloud{gceCloud: gce}
	errTwo := mockgce.CreateTCPLoadBalancer("tcpLoadBalancer", "us-central1-b", 10000, host)
	if errTwo == nil {
		t.Fatalf("Error expected")
	}
	errorMessage := "googleapi: Error 401: Login Required, required"
	if errTwo.Error() != errorMessage {
		t.Fatalf("Unexpected Error %v", errTwo)
	}
}
