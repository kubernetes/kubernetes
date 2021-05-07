// +build !providerless

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

	"github.com/GoogleCloudPlatform/k8s-cloud-provider/pkg/cloud"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	compute "google.golang.org/api/compute/v1"
)

const testSvcName = "my-service"
const testSubnet = "/projects/x/testRegions/us-central1/testSubnetworks/customsub"
const testLBName = "a111111111111111"

var vals = DefaultTestClusterValues()

// TestAddressManagerNoRequestedIP tests the typical case of passing in no requested IP
func TestAddressManagerNoRequestedIP(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	require.NoError(t, err)
	targetIP := ""

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, vals.Region, targetIP, string(cloud.SchemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerBasic tests the typical case of reserving and unreserving an address.
func TestAddressManagerBasic(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	require.NoError(t, err)
	targetIP := "1.1.1.1"

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, vals.Region, targetIP, string(cloud.SchemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerOrphaned tests the case where the address exists with the IP being equal
// to the requested address (forwarding rule or loadbalancer IP).
func TestAddressManagerOrphaned(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	require.NoError(t, err)
	targetIP := "1.1.1.1"

	addr := &compute.Address{Name: testLBName, Address: targetIP, AddressType: string(cloud.SchemeInternal)}
	err = svc.ReserveRegionAddress(addr, vals.Region)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, vals.Region, targetIP, string(cloud.SchemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerOutdatedOrphan tests the case where an address exists but points to
// an IP other than the forwarding rule or loadbalancer IP.
func TestAddressManagerOutdatedOrphan(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	require.NoError(t, err)
	previousAddress := "1.1.0.0"
	targetIP := "1.1.1.1"

	addr := &compute.Address{Name: testLBName, Address: previousAddress, AddressType: string(cloud.SchemeExternal)}
	err = svc.ReserveRegionAddress(addr, vals.Region)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, vals.Region, targetIP, string(cloud.SchemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerExternallyOwned tests the case where the address exists but isn't
// owned by the controller.
func TestAddressManagerExternallyOwned(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	require.NoError(t, err)
	targetIP := "1.1.1.1"

	addr := &compute.Address{Name: "my-important-address", Address: targetIP, AddressType: string(cloud.SchemeInternal)}
	err = svc.ReserveRegionAddress(addr, vals.Region)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	ipToUse, err := mgr.HoldAddress()
	require.NoError(t, err)
	assert.NotEmpty(t, ipToUse)

	ad, err := svc.GetRegionAddress(testLBName, vals.Region)
	assert.True(t, isNotFound(err))
	require.Nil(t, ad)

	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerExternallyOwned tests the case where the address exists but isn't
// owned by the controller. However, this address has the wrong type.
func TestAddressManagerBadExternallyOwned(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	require.NoError(t, err)
	targetIP := "1.1.1.1"

	addr := &compute.Address{Name: "my-important-address", Address: targetIP, AddressType: string(cloud.SchemeExternal)}
	err = svc.ReserveRegionAddress(addr, vals.Region)
	require.NoError(t, err)

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	ad, err := mgr.HoldAddress()
	assert.Error(t, err) // FIXME
	require.Equal(t, ad, "")
}

func testHoldAddress(t *testing.T, mgr *addressManager, svc CloudAddressService, name, region, targetIP, scheme string) {
	ipToUse, err := mgr.HoldAddress()
	require.NoError(t, err)
	assert.NotEmpty(t, ipToUse)

	addr, err := svc.GetRegionAddress(name, region)
	require.NoError(t, err)
	if targetIP != "" {
		assert.EqualValues(t, targetIP, addr.Address)
	}
	assert.EqualValues(t, scheme, addr.AddressType)
}

func testReleaseAddress(t *testing.T, mgr *addressManager, svc CloudAddressService, name, region string) {
	err := mgr.ReleaseAddress()
	require.NoError(t, err)
	_, err = svc.GetRegionAddress(name, region)
	assert.True(t, isNotFound(err))
}
