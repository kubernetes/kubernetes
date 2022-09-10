//go:build !providerless
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
	"github.com/google/go-cmp/cmp"
	compute "google.golang.org/api/compute/v1"
)

const testSvcName = "my-service"
const testSubnet = "/projects/x/testRegions/us-central1/testSubnetworks/customsub"
const testLBName = "a111111111111111"

var vals = DefaultTestClusterValues()

// TestAddressManagerNoRequestedIP tests the typical case of passing in no requested IP
func TestAddressManagerNoRequestedIP(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	if err != nil {
		t.Fatal(err)
	}
	targetIP := ""

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, vals.Region, targetIP, string(cloud.SchemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerBasic tests the typical case of reserving and unreserving an address.
func TestAddressManagerBasic(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	if err != nil {
		t.Fatal(err)
	}
	targetIP := "1.1.1.1"

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, vals.Region, targetIP, string(cloud.SchemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerOrphaned tests the case where the address exists with the IP being equal
// to the requested address (forwarding rule or loadbalancer IP).
func TestAddressManagerOrphaned(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	if err != nil {
		t.Fatal(err)
	}
	targetIP := "1.1.1.1"

	addr := &compute.Address{Name: testLBName, Address: targetIP, AddressType: string(cloud.SchemeInternal)}
	err = svc.ReserveRegionAddress(addr, vals.Region)
	if err != nil {
		t.Fatal(err)
	}

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, vals.Region, targetIP, string(cloud.SchemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerOutdatedOrphan tests the case where an address exists but points to
// an IP other than the forwarding rule or loadbalancer IP.
func TestAddressManagerOutdatedOrphan(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	if err != nil {
		t.Fatal(err)
	}
	previousAddress := "1.1.0.0"
	targetIP := "1.1.1.1"

	addr := &compute.Address{Name: testLBName, Address: previousAddress, AddressType: string(cloud.SchemeExternal)}
	err = svc.ReserveRegionAddress(addr, vals.Region)
	if err != nil {
		t.Fatal(err)
	}

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	testHoldAddress(t, mgr, svc, testLBName, vals.Region, targetIP, string(cloud.SchemeInternal))
	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerExternallyOwned tests the case where the address exists but isn't
// owned by the controller.
func TestAddressManagerExternallyOwned(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	if err != nil {
		t.Fatal(err)
	}
	targetIP := "1.1.1.1"

	addr := &compute.Address{Name: "my-important-address", Address: targetIP, AddressType: string(cloud.SchemeInternal)}
	err = svc.ReserveRegionAddress(addr, vals.Region)
	if err != nil {
		t.Fatal(err)
	}

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	ipToUse, err := mgr.HoldAddress()
	if err != nil {
		t.Fatal(err)
	}
	if len(ipToUse) == 0 {
		t.Error("ipToUse is empty")
	}

	ad, err := svc.GetRegionAddress(testLBName, vals.Region)
	if !cmp.Equal(true, isNotFound(err)) {
		t.Errorf("want: %t, got: %t", true, isNotFound(err))
	}
	if ad != nil {
		t.Error("ad is nil")
	}

	testReleaseAddress(t, mgr, svc, testLBName, vals.Region)
}

// TestAddressManagerExternallyOwned tests the case where the address exists but isn't
// owned by the controller. However, this address has the wrong type.
func TestAddressManagerBadExternallyOwned(t *testing.T) {
	svc, err := fakeGCECloud(vals)
	if err != nil {
		t.Fatal(err)
	}
	targetIP := "1.1.1.1"

	addr := &compute.Address{Name: "my-important-address", Address: targetIP, AddressType: string(cloud.SchemeExternal)}
	err = svc.ReserveRegionAddress(addr, vals.Region)
	if err != nil {
		t.Fatal(err)
	}

	mgr := newAddressManager(svc, testSvcName, vals.Region, testSubnet, testLBName, targetIP, cloud.SchemeInternal)
	ad, err := mgr.HoldAddress()
	if err == nil { // FIXME
		t.Error(err)
	}
	if ad != "" {
		t.Errorf("want: %s, got: %s", "", ad)
	}
}

func testHoldAddress(t *testing.T, mgr *addressManager, svc CloudAddressService, name, region, targetIP, scheme string) {
	ipToUse, err := mgr.HoldAddress()
	if err != nil {
		t.Fatal(err)
	}
	if ipToUse == "" {
		t.Error("ipToUse is not empty")
	}

	addr, err := svc.GetRegionAddress(name, region)
	if err != nil {
		t.Fatal(err)
	}
	if targetIP != "" {
		if targetIP != addr.Address {
			t.Errorf("want: %s, got: %s", targetIP, addr.Address)
		}
	}
	if scheme != addr.AddressType {
		t.Errorf("want: %s, got: %s", scheme, addr.AddressType)
	}
}

func testReleaseAddress(t *testing.T, mgr *addressManager, svc CloudAddressService, name, region string) {
	err := mgr.ReleaseAddress()
	if err != nil {
		t.Fatal(err)
	}
	_, err = svc.GetRegionAddress(name, region)
	if !cmp.Equal(true, isNotFound(err)) {
		t.Errorf("want: %t, got: %t", true, isNotFound(err))
	}
}
