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

package openstack

import (
	"reflect"
	"sort"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	"k8s.io/api/core/v1"
)

func TestInstanceIDFromProviderID(t *testing.T) {
	testCases := []struct {
		providerID string
		instanceID string
		fail       bool
	}{
		{
			providerID: ProviderName + "://" + "/" + "7b9cf879-7146-417c-abfd-cb4272f0c935",
			instanceID: "7b9cf879-7146-417c-abfd-cb4272f0c935",
			fail:       false,
		},
		{
			providerID: "openstack://7b9cf879-7146-417c-abfd-cb4272f0c935",
			instanceID: "",
			fail:       true,
		},
		{
			providerID: "7b9cf879-7146-417c-abfd-cb4272f0c935",
			instanceID: "",
			fail:       true,
		},
		{
			providerID: "other-provider:///7b9cf879-7146-417c-abfd-cb4272f0c935",
			instanceID: "",
			fail:       true,
		},
	}

	for _, test := range testCases {
		instanceID, err := instanceIDFromProviderID(test.providerID)
		if (err != nil) != test.fail {
			t.Errorf("%s yielded `err != nil` as %t. expected %t", test.providerID, (err != nil), test.fail)
		}

		if test.fail {
			continue
		}

		if instanceID != test.instanceID {
			t.Errorf("%s yielded %s. expected %s", test.providerID, instanceID, test.instanceID)
		}
	}
}

// An arbitrary sort.Interface, just for easier comparison
type AddressSlice []v1.NodeAddress

func (a AddressSlice) Len() int           { return len(a) }
func (a AddressSlice) Less(i, j int) bool { return a[i].Address < a[j].Address }
func (a AddressSlice) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

func TestNodeAddresses(t *testing.T) {
	srv := servers.Server{
		Status:     "ACTIVE",
		HostID:     "29d3c8c896a45aa4c34e52247875d7fefc3d94bbcc9f622b5d204362",
		AccessIPv4: "50.56.176.99",
		AccessIPv6: "2001:4800:790e:510:be76:4eff:fe04:82a8",
		Addresses: map[string]interface{}{
			"private": []interface{}{
				map[string]interface{}{
					"OS-EXT-IPS-MAC:mac_addr": "fa:16:3e:7c:1b:2b",
					"version":                 float64(4),
					"addr":                    "10.0.0.32",
					"OS-EXT-IPS:type":         "fixed",
				},
				map[string]interface{}{
					"version":         float64(4),
					"addr":            "50.56.176.36",
					"OS-EXT-IPS:type": "floating",
				},
				map[string]interface{}{
					"version": float64(4),
					"addr":    "10.0.0.31",
					// No OS-EXT-IPS:type
				},
			},
			"public": []interface{}{
				map[string]interface{}{
					"version": float64(4),
					"addr":    "50.56.176.35",
				},
				map[string]interface{}{
					"version": float64(6),
					"addr":    "2001:4800:780e:510:be76:4eff:fe04:84a8",
				},
			},
		},
	}

	addrs, err := nodeAddresses(&srv)
	if err != nil {
		t.Fatalf("nodeAddresses returned error: %v", err)
	}

	sort.Sort(AddressSlice(addrs))
	t.Logf("addresses is %v", addrs)

	want := []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: "10.0.0.31"},
		{Type: v1.NodeInternalIP, Address: "10.0.0.32"},
		{Type: v1.NodeExternalIP, Address: "2001:4800:780e:510:be76:4eff:fe04:84a8"},
		{Type: v1.NodeExternalIP, Address: "2001:4800:790e:510:be76:4eff:fe04:82a8"},
		{Type: v1.NodeExternalIP, Address: "50.56.176.35"},
		{Type: v1.NodeExternalIP, Address: "50.56.176.36"},
		{Type: v1.NodeExternalIP, Address: "50.56.176.99"},
	}

	if !reflect.DeepEqual(want, addrs) {
		t.Errorf("nodeAddresses returned incorrect value %v", addrs)
	}
}
