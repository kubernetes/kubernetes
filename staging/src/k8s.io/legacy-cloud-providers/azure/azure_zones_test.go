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

	"k8s.io/apimachinery/pkg/util/sets"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/legacy-cloud-providers/azure/clients/vmclient/mockvmclient"

	"github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute"
	"github.com/Azure/go-autorest/autorest/to"
	"github.com/golang/mock/gomock"
	"github.com/stretchr/testify/assert"
)

const (
	testAvailabilitySetNodeProviderID = "azure:///subscriptions/sub/resourceGroups/rg/providers/Microsoft.Compute/virtualMachines/vm-0"
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
		location    string
		faultDomain string
		expected    string
		isNilResp   bool
		expectedErr error
	}{
		{
			name:     "GetZone should get real zone if only node's zone is set",
			zone:     "1",
			location: "eastus",
			expected: "eastus-1",
		},
		{
			name:        "GetZone should get real zone if both node's zone and FD are set",
			zone:        "1",
			location:    "eastus",
			faultDomain: "99",
			expected:    "eastus-1",
		},
		{
			name:        "GetZone should get faultDomain if node's zone isn't set",
			location:    "eastus",
			faultDomain: "99",
			expected:    "99",
		},
		{
			name:     "GetZone should get availability zone in lower cases",
			location: "EastUS",
			zone:     "1",
			expected: "eastus-1",
		},
		{
			name:        "GetZone should report an error if there is no `Compute` in the response",
			isNilResp:   true,
			expectedErr: fmt.Errorf("failure of getting compute information from instance metadata"),
		},
		{
			name:        "GetZone should report an error if the zone is invalid",
			zone:        "a",
			location:    "eastus",
			faultDomain: "99",
			expected:    "",
			expectedErr: fmt.Errorf("failed to parse zone ID \"a\": strconv.Atoi: parsing \"a\": invalid syntax"),
		},
	}

	for _, test := range testcases {
		listener, err := net.Listen("tcp", "127.0.0.1:0")
		if err != nil {
			t.Errorf("Test [%s] unexpected error: %v", test.name, err)
		}

		respString := fmt.Sprintf(`{"compute":{"zone":"%s", "platformFaultDomain":"%s", "location":"%s"}}`, test.zone, test.faultDomain, test.location)
		if test.isNilResp {
			respString = "{}"
		}
		mux := http.NewServeMux()
		mux.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, respString)
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
			if test.expectedErr == nil {
				t.Errorf("Test [%s] unexpected error: %v", test.name, err)
			} else {
				assert.Equal(t, test.expectedErr, err)
			}
		}
		if zone.FailureDomain != test.expected {
			t.Errorf("Test [%s] unexpected zone: %s, expected %q", test.name, zone.FailureDomain, test.expected)
		}
		if err == nil && zone.Region != cloud.Location {
			t.Errorf("Test [%s] unexpected region: %s, expected: %s", test.name, zone.Region, cloud.Location)
		}
	}
}

func TestMakeZone(t *testing.T) {
	az := &Cloud{}
	zone := az.makeZone("EASTUS", 2)
	assert.Equal(t, "eastus-2", zone)
}

func TestGetZoneByProviderID(t *testing.T) {
	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	az := GetTestCloud(ctrl)

	zone, err := az.GetZoneByProviderID(context.Background(), "")
	assert.Equal(t, errNodeNotInitialized, err)
	assert.Equal(t, cloudprovider.Zone{}, zone)

	zone, err = az.GetZoneByProviderID(context.Background(), "invalid/id")
	assert.NoError(t, err)
	assert.Equal(t, cloudprovider.Zone{}, zone)

	mockVMClient := az.VirtualMachinesClient.(*mockvmclient.MockInterface)
	mockVMClient.EXPECT().Get(gomock.Any(), az.ResourceGroup, "vm-0", gomock.Any()).Return(compute.VirtualMachine{
		Zones:    &[]string{"1"},
		Location: to.StringPtr("eastus"),
	}, nil)
	zone, err = az.GetZoneByProviderID(context.Background(), testAvailabilitySetNodeProviderID)
	assert.NoError(t, err)
	assert.Equal(t, cloudprovider.Zone{
		FailureDomain: "eastus-1",
		Region:        "eastus",
	}, zone)
}

func TestAvailabilitySetGetZoneByNodeName(t *testing.T) {
	az := &Cloud{
		unmanagedNodes: sets.String{"vm-0": sets.Empty{}},
		nodeInformerSynced: func() bool {
			return true
		},
	}
	zone, err := az.GetZoneByNodeName(context.Background(), "vm-0")
	assert.NoError(t, err)
	assert.Equal(t, cloudprovider.Zone{}, zone)

	az = &Cloud{
		unmanagedNodes: sets.String{"vm-0": sets.Empty{}},
		nodeInformerSynced: func() bool {
			return false
		},
	}
	zone, err = az.GetZoneByNodeName(context.Background(), "vm-0")
	assert.Equal(t, fmt.Errorf("node informer is not synced when trying to GetUnmanagedNodes"), err)
	assert.Equal(t, cloudprovider.Zone{}, zone)
}
