// +build !providerless

/*
Copyright 2018 The Kubernetes Authors.

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

package azure

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"testing"
)

func TestIsAvailabilityZone(t *testing.T) {
	location := "eastus"
	az := &Cloud{
		Config: Config{
			Location: location,
		},
	}
	tests := []struct {
		desc     string
		zone     string
		expected bool
	}{
		{"empty string should return false", "", false},
		{"wrong farmat should return false", "123", false},
		{"wrong location should return false", "chinanorth-1", false},
		{"correct zone should return true", "eastus-1", true},
	}

	for _, test := range tests {
		actual := az.isAvailabilityZone(test.zone)
		if actual != test.expected {
			t.Errorf("test [%q] get unexpected result: %v != %v", test.desc, actual, test.expected)
		}
	}
}

func TestGetZoneID(t *testing.T) {
	location := "eastus"
	az := &Cloud{
		Config: Config{
			Location: location,
		},
	}
	tests := []struct {
		desc     string
		zone     string
		expected string
	}{
		{"empty string should return empty string", "", ""},
		{"wrong farmat should return empty string", "123", ""},
		{"wrong location should return empty string", "chinanorth-1", ""},
		{"correct zone should return zone ID", "eastus-1", "1"},
	}

	for _, test := range tests {
		actual := az.GetZoneID(test.zone)
		if actual != test.expected {
			t.Errorf("test [%q] get unexpected result: %q != %q", test.desc, actual, test.expected)
		}
	}
}

func TestGetZone(t *testing.T) {
	cloud := &Cloud{
		Config: Config{
			Location:            "eastus",
			UseInstanceMetadata: true,
		},
	}
	testcases := []struct {
		name        string
		zone        string
		faultDomain string
		expected    string
	}{
		{
			name:     "GetZone should get real zone if only node's zone is set",
			zone:     "1",
			expected: "eastus-1",
		},
		{
			name:        "GetZone should get real zone if both node's zone and FD are set",
			zone:        "1",
			faultDomain: "99",
			expected:    "eastus-1",
		},
		{
			name:        "GetZone should get faultDomain if node's zone isn't set",
			faultDomain: "99",
			expected:    "99",
		},
	}

	for _, test := range testcases {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Errorf("Test [%s] unexpected error: %v", test.name, err)
		}

		mux := http.NewServeMux()
		mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, fmt.Sprintf(`{"compute":{"zone":"%s", "platformFaultDomain":"%s", "location":"eastus"}}`, test.zone, test.faultDomain))
		}))
		go func() {
			http.Serve(listener, mux)
		}()
		defer listener.Close()

		cloud.metadata, err = NewInstanceMetadataService("http://" + listener.Addr().String() + "/")
		if err != nil {
			t.Errorf("Test [%s] unexpected error: %v", test.name, err)
		}

		zone, err := cloud.GetZone(context.Background())
		if err != nil {
			t.Errorf("Test [%s] unexpected error: %v", test.name, err)
		}
		if zone.FailureDomain != test.expected {
			t.Errorf("Test [%s] unexpected zone: %s, expected %q", test.name, zone.FailureDomain, test.expected)
		}
		if zone.Region != cloud.Location {
			t.Errorf("Test [%s] unexpected region: %s, expected: %s", test.name, zone.Region, cloud.Location)
		}
	}
}
