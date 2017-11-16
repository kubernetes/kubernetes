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

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	computebeta "google.golang.org/api/compute/v0.beta"
)

const testSvcName = "my-service"
const testRegion = "us-central1"
const testSubnet = "/projects/x/testRegions/us-central1/testSubnetworks/customsub"
const testLBName = "a111111111111111"

// TestAddressManagerNoRequestedIP tests the typical case of passing in no requested IP
func TestAddressManagerNoRequestedIP(t *testing.T) {
	svc := NewFakeCloudAddressService()
	targetIP := ""

	mgr := newAddressManager(svc, testSvcName, testRegion, testSubnet, testLBName, targetIP, schemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, testRegion, targetIP, string(schemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, testRegion)
}

// TestAddressManagerBasic tests the typical case of reserving and unreserving an address.
func TestAddressManagerBasic(t *testing.T) {
	svc := NewFakeCloudAddressService()
	targetIP := "1.1.1.1"

	mgr := newAddressManager(svc, testSvcName, testRegion, testSubnet, testLBName, targetIP, schemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, testRegion, targetIP, string(schemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, testRegion)
}

// TestAddressManagerOrphaned tests the case where the address exists with the IP being equal
// to the requested address (forwarding rule or loadbalancer IP).
func TestAddressManagerOrphaned(t *testing.T) {
	svc := NewFakeCloudAddressService()
	targetIP := "1.1.1.1"

	addr := &computebeta.Address{Name: testLBName, Address: targetIP, AddressType: string(schemeInternal)}
	err := svc.ReserveBetaRegionAddress(addr, testRegion)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testSvcName, testRegion, testSubnet, testLBName, targetIP, schemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, testRegion, targetIP, string(schemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, testRegion)
}

// TestAddressManagerOutdatedOrphan tests the case where an address exists but points to
// an IP other than the forwarding rule or loadbalancer IP.
func TestAddressManagerOutdatedOrphan(t *testing.T) {
	svc := NewFakeCloudAddressService()
	previousAddress := "1.1.0.0"
	targetIP := "1.1.1.1"

	addr := &computebeta.Address{Name: testLBName, Address: previousAddress, AddressType: string(schemeExternal)}
	err := svc.ReserveBetaRegionAddress(addr, testRegion)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testSvcName, testRegion, testSubnet, testLBName, targetIP, schemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, testRegion, targetIP, string(schemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, testRegion)
}

// TestAddressManagerExternallyOwned tests the case where the address exists but isn't
// owned by the controller.
func TestAddressManagerExternallyOwned(t *testing.T) {
	svc := NewFakeCloudAddressService()
	targetIP := "1.1.1.1"

	addr := &computebeta.Address{Name: "my-important-address", Address: targetIP, AddressType: string(schemeInternal)}
	err := svc.ReserveBetaRegionAddress(addr, testRegion)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testSvcName, testRegion, testSubnet, testLBName, targetIP, schemeInternal)
	ipToUse, err := mgr.HoldAddress()
	require.NoError(t, err)
	assert.NotEmpty(t, ipToUse)

	_, err = svc.GetRegionAddress(testLBName, testRegion)
	assert.True(t, isNotFound(err))

	testReleaseAddress(t, mgr, svc, testLBName, testRegion)
}

// TestAddressManagerExternallyOwned tests the case where the address exists but isn't
// owned by the controller. However, this address has the wrong type.
func TestAddressManagerBadExternallyOwned(t *testing.T) {
	svc := NewFakeCloudAddressService()
	targetIP := "1.1.1.1"

	addr := &computebeta.Address{Name: "my-important-address", Address: targetIP, AddressType: string(schemeExternal)}
	err := svc.ReserveBetaRegionAddress(addr, testRegion)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testSvcName, testRegion, testSubnet, testLBName, targetIP, schemeInternal)
	_, err = mgr.HoldAddress()
	assert.NotNil(t, err)
}

func testHoldAddress(t *testing.T, mgr *addressManager, svc CloudAddressService, name, region, targetIP, scheme string) {
	ipToUse, err := mgr.HoldAddress()
	require.NoError(t, err)
	assert.NotEmpty(t, ipToUse)

	addr, err := svc.GetBetaRegionAddress(name, region)
	require.NoError(t, err)
	if targetIP != "" {
		assert.EqualValues(t, targetIP, addr.Address)
	}
	assert.EqualValues(t, scheme, addr.AddressType)
}

func testReleaseAddress(t *testing.T, mgr *addressManager, svc CloudAddressService, name, region string) {
	err := mgr.ReleaseAddress()
	require.NoError(t, err)
	_, err = svc.GetBetaRegionAddress(name, region)
	assert.True(t, isNotFound(err))
}
