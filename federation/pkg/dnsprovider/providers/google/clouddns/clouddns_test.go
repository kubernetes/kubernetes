/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package clouddns

import (
	"fmt"
	"testing"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"

	dns "google.golang.org/api/dns/v1"
)

type (
	fakeChangesService struct {
	}

	fakeManagedZonesService struct {
	}

	fakeManagedZonesListCall struct {
	}

	fakeManagedZonesListResponse struct {
	}

	fakeProjectsService struct {
	}

	fakeResourceRecordSetsService struct {
	}

	fakeIface struct {
		fakeService
	}

	fakeService struct {
		basePath           string // API endpoint base URL
		userAgent          string // optional additional User-Agent fragment
		changes            fakeChangesService
		managedZones       fakeManagedZonesService
		projects           fakeProjectsService
		resourceRecordSets fakeResourceRecordSetsService
	}
)

func newFakeInterface() iface {
	/*
		var srv service = fakeService{"", "", fakeChangesService{}, fakeManagedZonesService{}, fakeProjectsService{}, fakeResourceRecordSetsService{}}
		var i iface = fakeIface{srv}
		return iface
	*/
	return NewInterface()
}

func (f fakeManagedZonesListCall) Do() (*fakeManagedZonesListResponse, error) {
	return &fakeManagedZonesListResponse{}, nil
}

func (f fakeManagedZonesListResponse) ManagedZones() []*dns.ManagedZone {
	fmt.Println("Getting Managed Zones")
	return []*dns.ManagedZone{}
}

func (z fakeManagedZonesService) List() *fakeManagedZonesListCall {
	return &fakeManagedZonesListCall{}
}

/* firstZone returns the first zone for the configured dns provider account/project, or fails if it can't be found */
func firstZone(t *testing.T) dnsprovider.Zone {
	i := newFakeInterface()
	t.Logf("Getting zones")
	z, supported := i.Zones()
	if supported {
		t.Logf("Got zones %v\n", z)
	} else {
		t.Errorf("Zones interface not supported by interface %v", i)
	}
	zones, err := z.List()
	if err != nil {
		t.Errorf("Failed to list zones: %v", err)
	} else {
		t.Logf("Got zone list: %v\n", zones)
	}
	if len(zones) < 1 {
		t.Errorf("Zone listing returned %d, expected %d", len(zones), 1)
	} else {
		t.Logf("Got at least 1 zone in list:%v\n", zones[0])
	}
	return zones[0]
}

func TestZonesList(t *testing.T) {
	firstZone(t)
}

/* rrs returns the ResourceRecordSets interface for a given zone */
func rrs(t *testing.T, zone dnsprovider.Zone) (r dnsprovider.ResourceRecordSets) {
	rrsets, supported := zone.ResourceRecordSets()
	if !supported {
		t.Errorf("ResourceRecordSets interface not supported by zone %v", zone)
		return r
	}
	return rrsets
}

/* TestResourceRecordSetsList verifies that listing of RRS's succeeds */
func TestResourceRecordSetsList(t *testing.T) {
	zone := firstZone(t)
	rrsets := rrs(t, zone)
	rrset, err := rrsets.List()
	if err != nil {
		t.Errorf("Failed to list recordsets: %v", err)
	}
	if len(rrset) < 1 {
		t.Errorf("Record set length=%d, expected >0", len(rrset))
	} else {
		t.Logf("Got %d recordsets: %v", len(rrset), rrset)
	}
}

/* TestResourceRecordSetsAddSuccess verifies that addition of a valid RRS succeeds */
func TestResourceRecordSetsAddSuccess(t *testing.T) {
	zone := firstZone(t)
	rrsets := rrs(t, zone)
	rrsToAdd := rrsets.New("www9."+zone.Name(), []string{"10.10.10.10", "20.20.20.20"}, 180, rrstype.A)
	rrs, err := rrsets.Add(rrsToAdd)
	if err != nil {
		t.Errorf("Failed to add resource record set %v: %v", rrsToAdd, err)
	} else {
		defer rrsets.Remove(rrs)
		t.Logf("Successfully added resource record set: %v", rrs)
	}
}

/* TestResourceRecordSetsAdditionVisible verifies that added RRS is visible after addition */
func TestResourceRecordSetsAdditionVisible(t *testing.T) {
	zone := firstZone(t)
	rrsets := rrs(t, zone)
	rrsToAdd := rrsets.New("www9."+zone.Name(), []string{"10.10.10.10", "20.20.20.20"}, 180, rrstype.A)
	rrs, err := rrsets.Add(rrsToAdd)
	if err != nil {
		t.Errorf("Failed to add resource record set %v: %v", rrsToAdd, err)
	} else {
		defer rrsets.Remove(rrs)
		t.Logf("Successfully added resource record set: %v", rrs)
	}
	list, err := rrsets.List()
	if err != nil {
		t.Errorf("Failed to list resource record sets for zone %s: %v", zone.Name(), err)
	}
	found := false
	for _, record := range list {
		if record.Name() == "www9."+zone.Name() {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("Failed to list added resource record set %v: %v", rrs, err)
	}
}

/* TestResourceRecordSetsAddDuplicateFail verifies that addition of a duplicate RRS fails */
func TestResourceRecordSetsAddDuplicateFail(t *testing.T) {
	zone := firstZone(t)
	rrsets := rrs(t, zone)
	rrsToAdd := rrsets.New("www9."+zone.Name(), []string{"10.10.10.10", "20.20.20.20"}, 180, rrstype.A)
	rrs, err := rrsets.Add(rrsToAdd)
	if err != nil {
		t.Errorf("Failed to add resource record set %v: %v", rrsToAdd, err)
	} else {
		defer rrsets.Remove(rrs)
		t.Logf("Successfully added resource record set: %v", rrsToAdd)
	}
	// Try to add it again, and verify that the call fails.
	rrs, err = rrsets.Add(rrsToAdd)
	if err == nil {
		defer rrsets.Remove(rrsToAdd)
		t.Logf("Successfully added duplicate resource record %v", rrsToAdd)
		// Spew out some additional info to aid debugging.
		records, _ := rrsets.List()
		for _, rr := range records {
			t.Logf("resource record set: %s, %d, %s", rr.Name(), rr.Ttl(), rr.Type())
			for _, data := range rr.Rrdatas() {
				t.Logf("%s", data)
			}
		}
		t.Errorf("Should have failed to add duplicate resource record %v, but succeeded instead.", rrsToAdd)
	} else {
		t.Logf("Correctly failed to add duplicate resource record %v: %v", rrsToAdd, err)
	}
}

/* TestResourceRecordSetsAddInvalidFail verifies that addition of an invalid RRS fails */
func TestResourceRecordSetsAddInvalidFail(t *testing.T) {
	zone := firstZone(t)
	rrsets := rrs(t, zone)
	rrsToAdd := rrsets.New("......"+zone.Name(), []string{"rubbish", "rubbish"}, 180, rrstype.A)
	rrs, err := rrsets.Add(rrsToAdd)
	if err == nil {
		defer rrsets.Remove(rrsToAdd)
		t.Errorf("Successfully added invalid resource record set %v.  Should have failed.", rrs)
	} else {
		t.Logf("Correctly failed to add invalid resource record set: %v", rrsToAdd)
	}
}

/* TestResourceRecordSetsRemove verifies that the removal of an existing RRS succeeds */
func TestResourceRecordSetsRemove(t *testing.T) {
	zone := firstZone(t)
	rrsets := rrs(t, zone)
	rrsToAdd := rrsets.New("www9."+zone.Name(), []string{"10.10.10.10", "20.20.20.20"}, 180, rrstype.A)
	rrs, err := rrsets.Add(rrsToAdd)
	if err != nil {
		t.Errorf("Failed to add resource record set %v: %v", rrsToAdd, err)
	} else {
		defer rrsets.Remove(rrs)
		t.Logf("Successfully added resource record set: %v", rrs)
	}
	err = rrsets.Remove(rrs)
	if err != nil {
		// Try again to clean up.
		defer rrsets.Remove(rrsets.New("www9."+zone.Name(), []string{"10.10.10.10", "20.20.20.20"}, 180, rrstype.A))
		t.Error("Failed to remove resource record set %v after adding", rrs)
	} else {
		t.Logf("Successfully removed resource set %v after adding", rrs)
	}
}

/* TestResourceRecordSetsRemoveGone verifies that a removed RRS no longer exists */
func TestResourceRecordSetsRemoveGone(t *testing.T) {
	zone := firstZone(t)
	rrsets := rrs(t, zone)
	rrsToAdd := rrsets.New("www9."+zone.Name(), []string{"10.10.10.10", "20.20.20.20"}, 180, rrstype.A)
	rrs, err := rrsets.Add(rrsToAdd)
	if err != nil {
		t.Errorf("Failed to add resource record set %v: %v", rrsToAdd, err)
	} else {
		defer rrsets.Remove(rrs)
		t.Logf("Successfully added resource record set: %v", rrs)
	}
	err = rrsets.Remove(rrs)
	if err != nil {
		// Try again to clean up.
		defer rrsets.Remove(rrsets.New("www9."+zone.Name(), []string{"10.10.10.10", "20.20.20.20"}, 180, rrstype.A))
		t.Error("Failed to remove resource record set %v after adding", rrs)
	} else {
		t.Logf("Successfully removed resource set %v after adding", rrs)
	}
	// Check that it's gone
	list, err := rrsets.List()
	if err != nil {
		t.Errorf("Failed to list resource record sets for zone %s: %v", zone.Name(), err)
	}
	found := false
	for _, record := range list {
		if record.Name() == rrs.Name() {
			found = true
			break
		}
	}
	if found {
		t.Errorf("Deleted resource record set %v is still present", rrs)
	}
}
