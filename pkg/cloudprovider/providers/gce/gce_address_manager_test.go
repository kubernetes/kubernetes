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
	"testing"

	"github.com/stretchr/testify/require"
	computealpha "google.golang.org/api/compute/v0.alpha"
	compute "google.golang.org/api/compute/v1"
)

const testSvcName = "my-service"
const testRegion = "us-central1"
const testSubnet = "/projects/x/testRegions/us-central1/testSubnetworks/customsub"
const testLBName = "a1234"

func newInternalAddr(requestedIP, currentIP string) addressParams {
	return addressParams{
		name:        testLBName,
		requestedIP: requestedIP,
		currentIP:   currentIP,
		addressType: schemeInternal,
	}
}

func newExternalAddr(requestedIP, currentIP string, tier NetworkTier) addressParams {
	return addressParams{
		name:        testLBName,
		requestedIP: requestedIP,
		currentIP:   currentIP,
		addressType: schemeExternal,
		netTier:     tier,
	}
}

func TestAddressManager(t *testing.T) {
	t.Parallel()

	alphaFeatureGate, err := NewAlphaFeatureGate([]string{AlphaFeatureNetworkTiers})
	require.NoError(t, err)

	tc := map[string]struct {
		params             addressParams
		existingAddrs      []compute.Address
		existingAlphaAddrs []computealpha.Address

		expectedIP   string
		expectedType lbScheme
		expectedErr  bool
	}{
		// Internal Addresses
		"create ILB": {
			params:       newInternalAddr("", ""),
			expectedIP:   "", // whatever is assigned
			expectedType: schemeInternal,
		},
		"sync ILB": {
			params:       newInternalAddr("", "1.1.2.1"),
			expectedIP:   "1.1.2.1",
			expectedType: schemeInternal,
		},
		"create ILB - requested IP": {
			params:       newInternalAddr("1.1.1.1", ""),
			expectedIP:   "1.1.1.1",
			expectedType: schemeInternal,
		},
		"sync ILB - requested IP": {
			params:       newInternalAddr("1.1.1.1", "1.1.1.1"),
			expectedIP:   "1.1.1.1",
			expectedType: schemeInternal,
		},
		"sync ILB - change requested IP": {
			params:       newInternalAddr("2.2.2.2", "1.1.1.1"),
			expectedIP:   "2.2.2.2",
			expectedType: schemeInternal,
		},

		// External Addresses
		"create ELB": {
			params:       newExternalAddr("", "", NetworkTierDefault),
			expectedType: schemeExternal,
		},
		"create ELB - Standard": {
			params:       newExternalAddr("", "", NetworkTierStandard),
			expectedType: schemeExternal,
		},
		"sync ELB": {
			params:       newExternalAddr("", "35.35.35.35", NetworkTierDefault),
			expectedIP:   "35.35.35.35",
			expectedType: schemeExternal,
		},
		"sync ELB - requested IP": {
			params:       newExternalAddr("35.35.35.35", "35.35.35.35", NetworkTierStandard),
			expectedIP:   "35.35.35.35",
			expectedType: schemeExternal,
		},

		// Controller owned orphan cases
		"sync ILB - controller owned addr exists": {
			params:        newInternalAddr("", "1.1.2.1"),
			existingAddrs: []compute.Address{{Name: testLBName, Address: "1.1.2.1", AddressType: string(schemeInternal)}},
			expectedIP:    "1.1.2.1",
			expectedType:  schemeInternal,
		},
		"sync ILB - controller owned addr exists with wrong IP": {
			params:        newInternalAddr("", "1.1.2.1"),
			existingAddrs: []compute.Address{{Name: testLBName, Address: "3.3.3.3", AddressType: string(schemeInternal)}},
			expectedIP:    "1.1.2.1",
			expectedType:  schemeInternal,
		},
		"sync ILB - controller owned addr exists with wrong scheme": {
			params:        newInternalAddr("", "1.1.2.1"),
			existingAddrs: []compute.Address{{Name: testLBName, Address: "35.35.35.35", AddressType: string(schemeExternal)}},
			expectedIP:    "1.1.2.1",
			expectedType:  schemeInternal,
		},

		// User owned address case
		"create ILB - user owned addr exists": {
			params:        newInternalAddr("1.2.3.4", ""),
			existingAddrs: []compute.Address{{Name: "my-addr", Address: "1.2.3.4", AddressType: string(schemeInternal)}},
			expectedIP:    "1.2.3.4",
			expectedType:  schemeInternal,
		},
		"sync ILB - user owned addr exists": {
			params:        newInternalAddr("", "1.2.3.4"),
			existingAddrs: []compute.Address{{Name: "my-addr", Address: "1.2.3.4", AddressType: string(schemeInternal)}},
			expectedIP:    "1.2.3.4",
			expectedType:  schemeInternal,
		},

		"create ELB - user owned addr requested": {
			params:        newExternalAddr("35.35.35.35", "", NetworkTierDefault),
			existingAddrs: []compute.Address{{Name: "my-addr", Address: "35.35.35.35", AddressType: string(schemeExternal)}},
			expectedIP:    "35.35.35.35",
			expectedType:  schemeExternal,
		},
		"sync ELB - user owned addr requested": {
			params:        newExternalAddr("35.35.35.35", "35.35.35.35", NetworkTierDefault),
			existingAddrs: []compute.Address{{Name: "my-addr", Address: "35.35.35.35", AddressType: string(schemeExternal)}},
			expectedIP:    "35.35.35.35",
			expectedType:  schemeExternal,
		},

		"sync ELB - user owned addr exists": {
			params:        newExternalAddr("", "35.35.35.35", NetworkTierDefault),
			existingAddrs: []compute.Address{{Name: "my-addr", Address: "35.35.35.35", AddressType: string(schemeExternal)}},
			expectedIP:    "35.35.35.35",
			expectedType:  schemeExternal,
		},

		"create ELB - user requested addr exists with wrong tier": {
			params:             newExternalAddr("35.35.35.35", "", NetworkTierDefault),
			existingAlphaAddrs: []computealpha.Address{{Name: "my-addr", Address: "35.35.35.35", AddressType: string(schemeExternal), NetworkTier: NetworkTierStandard.ToGCEValue()}},
			expectedErr:        true,
		},
	}

	for name, c := range tc {
		t.Run(name, func(t *testing.T) {
			svc := NewFakeCloudAddressService()

			// Reserve existing addresses
			for _, a := range c.existingAddrs {
				x := a
				err := svc.ReserveRegionAddress(&x, testRegion)
				require.NoError(t, err)
			}
			for _, a := range c.existingAlphaAddrs {
				x := a
				err := svc.ReserveAlphaRegionAddress(&x, testRegion)
				require.NoError(t, err)
			}

			// Hold address
			mgr := newAddressManager(svc, testRegion, testSubnet, c.params, alphaFeatureGate)
			gotIP, err := mgr.HoldAddress()

			if c.expectedErr != (err != nil) {
				t.Fatal(err)
			}

			if c.expectedErr {
				return
			}

			// Assert IP exists and is as expected
			require.NotEmpty(t, gotIP)
			if c.expectedIP != "" {
				require.Equal(t, c.expectedIP, gotIP)
			}

			// Check something is reserving the IP
			addr, err := svc.GetRegionAddressByIP(testRegion, gotIP)
			require.NoError(t, err)
			require.EqualValues(t, string(c.expectedType), addr.AddressType)

			expectedNetTier := c.params.netTier.ToGCEValue()
			if c.params.addressType == schemeExternal && expectedNetTier == "" {
				expectedNetTier = NetworkTierDefault.ToGCEValue()
			}

			// Assert network tier
			alphaAddr, _ := svc.GetAlphaRegionAddress(addr.Name, testRegion)
			require.EqualValues(t, expectedNetTier, alphaAddr.NetworkTier)

			// Release address (this is a no-op if user-owned)
			err = mgr.ReleaseAddress()
			require.NoError(t, err)

			// Check controller no longer owns an address.
			_, err = svc.GetRegionAddress(testLBName, testRegion)
			require.True(t, isNotFound(err))

			// Assert that existing addresses (that aren't owned by the controller)
			// continue to exist.
			for _, a := range c.existingAddrs {
				if a.Name == testLBName {
					continue
				}
				_, err := svc.GetRegionAddress(a.Name, testRegion)
				require.NoError(t, err)
			}
			for _, a := range c.existingAlphaAddrs {
				if a.Name == testLBName {
					continue
				}
				_, err := svc.GetRegionAddress(a.Name, testRegion)
				require.NoError(t, err)
			}
		})
	}
}

// TestAddressManagerRequestWrongType tests the case where the requestedIP is reserved by the user
// but is the wrong type.
func TestAddressManagerRequestWrongType(t *testing.T) {
	svc := NewFakeCloudAddressService()
	requestedIP := "35.35.35.35"

	addr := &compute.Address{Name: "my-addr", Address: requestedIP, AddressType: string(schemeExternal)}
	err := svc.ReserveRegionAddress(addr, testRegion)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testRegion, testSubnet, newInternalAddr(requestedIP, ""), nil)
	_, err = mgr.HoldAddress()
	require.Error(t, err)
}
