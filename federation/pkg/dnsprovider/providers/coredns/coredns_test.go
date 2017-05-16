/*
Copyright 2016 The Kubernetes Authors.

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

package coredns

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"testing"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	corednstesting "k8s.io/kubernetes/federation/pkg/dnsprovider/providers/coredns/stubs"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"

	"k8s.io/kubernetes/federation/pkg/dnsprovider/tests"
	"strings"
)

func newTestInterface() (dnsprovider.Interface, error) {
	// Use this to test the real cloud service.
	// return dnsprovider.GetDnsProvider(ProviderName, strings.NewReader("\n[global]\nproject-id = federation0-cluster00"))
	return newFakeInterface() // Use this to stub out the entire cloud service
}

func newFakeInterface() (dnsprovider.Interface, error) {
	var service corednstesting.EtcdKeysAPI
	service = corednstesting.NewEtcdKeysAPIStub()
	intf := newInterfaceWithStub(service)
	intf.etcdPathPrefix = "skydns"

	zoneList := strings.Split("example.com,federation.io", ",")
	intf.zones = Zones{intf: intf}
	for index, zoneName := range zoneList {
		zone := Zone{domain: zoneName, id: strconv.Itoa(index), zones: &intf.zones}
		intf.zones.zoneList = append(intf.zones.zoneList, zone)
	}

	return intf, nil
}

var intf dnsprovider.Interface

func TestMain(m *testing.M) {
	fmt.Printf("Parsing flags.\n")
	flag.Parse()
	var err error
	fmt.Printf("Getting new test interface.\n")
	intf, err = newTestInterface()
	if err != nil {
		fmt.Printf("Error creating interface: %v", err)
		os.Exit(1)
	}
	fmt.Printf("Running tests...\n")
	os.Exit(m.Run())
}

// zones returns the zones interface for the configured dns provider account/project,
// or fails if it can't be found
func zones(t *testing.T) dnsprovider.Zones {
	zonesInterface, supported := intf.Zones()
	if !supported {
		t.Fatalf("Zones interface not supported by interface %v", intf)
	} else {
		t.Logf("Got zones %v\n", zonesInterface)
	}
	return zonesInterface
}

// firstZone returns the first zone for the configured dns provider account/project,
// or fails if it can't be found
func firstZone(t *testing.T) dnsprovider.Zone {
	t.Logf("Getting zones")
	z := zones(t)
	zones, err := z.List()
	if err != nil {
		t.Fatalf("Failed to list zones: %v", err)
	} else {
		t.Logf("Got zone list: %v\n", zones)
	}
	if len(zones) < 1 {
		t.Fatalf("Zone listing returned %d, expected >= %d", len(zones), 1)
	} else {
		t.Logf("Got at least 1 zone in list:%v\n", zones[0])
	}
	return zones[0]
}

/* rrs returns the ResourceRecordSets interface for a given zone */
func rrs(t *testing.T, zone dnsprovider.Zone) (r dnsprovider.ResourceRecordSets) {
	rrsets, supported := zone.ResourceRecordSets()
	if !supported {
		t.Fatalf("ResourceRecordSets interface not supported by zone %v", zone)
		return r
	}
	return rrsets
}

func listRrsOrFail(t *testing.T, rrsets dnsprovider.ResourceRecordSets) []dnsprovider.ResourceRecordSet {
	rrset, err := rrsets.List()
	if err != nil {
		t.Fatalf("Failed to list recordsets: %v", err)
	} else {
		if len(rrset) < 0 {
			t.Fatalf("Record set length=%d, expected >=0", len(rrset))
		} else {
			t.Logf("Got %d recordsets: %v", len(rrset), rrset)
		}
	}
	return rrset
}

func getRrOrFail(t *testing.T, rrsets dnsprovider.ResourceRecordSets, name string) []dnsprovider.ResourceRecordSet {
	rrsetList, err := rrsets.Get(name)
	if err != nil {
		t.Fatalf("Failed to get recordset: %v", err)
	} else if len(rrsetList) == 0 {
		t.Logf("Did not Get recordset: %v", name)
	} else {
		t.Logf("Got recordsets: %v", rrsetList)
	}
	return rrsetList
}

func getExampleRrs(zone dnsprovider.Zone) dnsprovider.ResourceRecordSet {
	rrsets, _ := zone.ResourceRecordSets()
	return rrsets.New("www11."+zone.Name(), []string{"10.10.10.10", "169.20.20.20"}, 180, rrstype.A)
}

func addRrsetOrFail(t *testing.T, rrsets dnsprovider.ResourceRecordSets, rrset dnsprovider.ResourceRecordSet) {
	err := rrsets.StartChangeset().Add(rrset).Apply()
	if err != nil {
		t.Fatalf("Failed to add recordsets: %v", err)
	}
}

/* TestZonesList verifies that listing of zones succeeds */
func TestZonesList(t *testing.T) {
	firstZone(t)
}

/* TestZonesID verifies that the id of the zone is unique */
func TestZonesID(t *testing.T) {
	zone := firstZone(t)

	zoneID := zone.ID()
	if zoneID != "0" {
		t.Fatalf("Unexpected zone id: %q", zoneID)
	}
}

/* TestResourceRecordSetsGet verifies that getting of RRS succeeds */
func TestResourceRecordSetsGet(t *testing.T) {
	getRrOrFail(t, rrs(t, firstZone(t)), "example.com")
}

/* TestResourceRecordSetsAddSuccess verifies that addition of a valid RRS succeeds */
func TestResourceRecordSetsAddSuccess(t *testing.T) {
	zone := firstZone(t)
	sets := rrs(t, zone)
	set := getExampleRrs(zone)
	addRrsetOrFail(t, sets, set)
	defer sets.StartChangeset().Remove(set).Apply()
	t.Logf("Successfully added resource record set: %v", set)
	if sets.Zone().ID() != zone.ID() {
		t.Errorf("Zone for rrset does not match expected")
	}
}

/* TestResourceRecordSetsAdditionVisible verifies that added RRS is visible after addition */
func TestResourceRecordSetsAdditionVisible(t *testing.T) {
	zone := firstZone(t)
	sets := rrs(t, zone)
	rrset := getExampleRrs(zone)
	addRrsetOrFail(t, sets, rrset)
	defer sets.StartChangeset().Remove(rrset).Apply()
	t.Logf("Successfully added resource record set: %v", rrset)

	record := getRrOrFail(t, sets, rrset.Name())
	if record == nil {
		t.Errorf("Failed to find added resource record set %s", rrset.Name())
	}
}

/* TestResourceRecordSetsAddDuplicateFail verifies that addition of a duplicate RRS fails */
func TestResourceRecordSetsAddDuplicateFail(t *testing.T) {
	zone := firstZone(t)
	sets := rrs(t, zone)
	rrset := getExampleRrs(zone)
	addRrsetOrFail(t, sets, rrset)
	defer sets.StartChangeset().Remove(rrset).Apply()
	t.Logf("Successfully added resource record set: %v", rrset)
	// Try to add it again, and verify that the call fails.
	err := sets.StartChangeset().Add(rrset).Apply()
	if err == nil {
		defer sets.StartChangeset().Remove(rrset).Apply()
		t.Errorf("Should have failed to add duplicate resource record %v, but succeeded instead.", rrset)
	} else {
		t.Logf("Correctly failed to add duplicate resource record %v: %v", rrset, err)
	}
}

/* TestResourceRecordSetsRemove verifies that the removal of an existing RRS succeeds */
func TestResourceRecordSetsRemove(t *testing.T) {
	zone := firstZone(t)
	sets := rrs(t, zone)
	rrset := getExampleRrs(zone)
	addRrsetOrFail(t, sets, rrset)
	err := sets.StartChangeset().Remove(rrset).Apply()
	if err != nil {
		// Try again to clean up.
		defer sets.StartChangeset().Remove(rrset).Apply()
		t.Errorf("Failed to remove resource record set %v after adding", rrset)
	} else {
		t.Logf("Successfully removed resource set %v after adding", rrset)
	}
}

/* TestResourceRecordSetsRemoveGone verifies that a removed RRS no longer exists */
func TestResourceRecordSetsRemoveGone(t *testing.T) {
	zone := firstZone(t)
	sets := rrs(t, zone)
	rrset := getExampleRrs(zone)
	addRrsetOrFail(t, sets, rrset)
	err := sets.StartChangeset().Remove(rrset).Apply()
	if err != nil {
		// Try again to clean up.
		defer sets.StartChangeset().Remove(rrset).Apply()
		t.Errorf("Failed to remove resource record set %v after adding", rrset)
	} else {
		t.Logf("Successfully removed resource set %v after adding", rrset)
	}

	record := getRrOrFail(t, sets, rrset.Name())
	if record != nil {
		t.Errorf("Deleted resource record set %v is still present", rrset)
	}
}

/* TestResourceRecordSetsReplace verifies that replacing an RRS works */
func TestResourceRecordSetsReplace(t *testing.T) {
	zone := firstZone(t)
	tests.CommonTestResourceRecordSetsReplace(t, zone)
}

/* TestResourceRecordSetsReplaceAll verifies that we can remove an RRS and create one with a different name */
func TestResourceRecordSetsReplaceAll(t *testing.T) {
	zone := firstZone(t)
	tests.CommonTestResourceRecordSetsReplaceAll(t, zone)
}

/* TestResourceRecordSetsDifferentTypes verifies that we can add records with same name, but different types */
func TestResourceRecordSetsDifferentTypes(t *testing.T) {
	zone := firstZone(t)
	tests.CommonTestResourceRecordSetsDifferentTypes(t, zone)
}
