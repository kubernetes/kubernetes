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

package tests

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/rrstype"
)

/* CommonTestResourceRecordSetsReplace verifies that replacing an RRS works */
func CommonTestResourceRecordSetsReplace(t *testing.T, zone dnsprovider.Zone) {
	rrsets, _ := zone.ResourceRecordSets()

	sets := rrs(t, zone)
	rrset := rrsets.New("alpha.test.com", []string{"8.8.4.4"}, 40, rrstype.A)
	addRrsetOrFail(t, sets, rrset)
	defer sets.StartChangeset().Remove(rrset).Apply()
	t.Logf("Successfully added resource record set: %v", rrset)

	// Replace the record (change ttl and rrdatas)
	newRrset := rrsets.New("alpha.test.com", []string{"8.8.8.8"}, 80, rrstype.A)
	err := sets.StartChangeset().Add(newRrset).Remove(rrset).Apply()
	if err != nil {
		t.Errorf("Failed to replace resource record set %v -> %v: %v", rrset, newRrset, err)
	} else {
		t.Logf("Correctly replaced resource record %v -> %v", rrset, newRrset)
	}
	defer sets.StartChangeset().Remove(newRrset).Apply()

	// Check that the record was updated
	assertHasRecord(t, sets, newRrset)
}

/* CommonTestResourceRecordSetsReplaceAll verifies that we can remove an RRS and create one with a different name*/
func CommonTestResourceRecordSetsReplaceAll(t *testing.T, zone dnsprovider.Zone) {
	rrsets, _ := zone.ResourceRecordSets()

	sets := rrs(t, zone)
	rrset := rrsets.New("alpha.test.com", []string{"8.8.4.4"}, 40, rrstype.A)
	addRrsetOrFail(t, sets, rrset)
	defer sets.StartChangeset().Remove(rrset).Apply()
	t.Logf("Successfully added resource record set: %v", rrset)

	newRrset := rrsets.New("beta.test.com", []string{"8.8.8.8"}, 80, rrstype.A)

	// Try to add it again, and verify that the call fails.
	err := sets.StartChangeset().Add(newRrset).Remove(rrset).Apply()
	if err != nil {
		t.Errorf("Failed to replace resource record set %v -> %v: %v", rrset, newRrset, err)
	} else {
		defer sets.StartChangeset().Remove(newRrset).Apply()
		t.Logf("Correctly replaced resource record %v -> %v", rrset, newRrset)
	}

	// Check that it was updated
	assertHasRecord(t, sets, newRrset)
	assertNotHasRecord(t, sets, rrset.Name(), rrset.Type())
}

/* CommonTestResourceRecordSetsHonorsType verifies that we can add records of the same name but different types */
func CommonTestResourceRecordSetsDifferentTypes(t *testing.T, zone dnsprovider.Zone) {
	rrsets, _ := zone.ResourceRecordSets()

	sets := rrs(t, zone)
	rrset := rrsets.New("alpha.test.com", []string{"8.8.4.4"}, 40, rrstype.A)
	addRrsetOrFail(t, sets, rrset)
	defer sets.StartChangeset().Remove(rrset).Apply()
	t.Logf("Successfully added resource record set: %v", rrset)

	aaaaRrset := rrsets.New("alpha.test.com", []string{"2001:4860:4860::8888"}, 80, rrstype.AAAA)

	// Add the resource with the same name but different type
	err := sets.StartChangeset().Add(aaaaRrset).Apply()
	if err != nil {
		t.Errorf("Failed to add resource record set %v: %v", aaaaRrset, err)
	}
	defer sets.StartChangeset().Remove(aaaaRrset).Apply()

	// Check that both records exist
	assertHasRecord(t, sets, aaaaRrset)
	assertHasRecord(t, sets, rrset)
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

// assertHasRecord tests that rrsets has a record equivalent to rrset
func assertHasRecord(t *testing.T, rrsets dnsprovider.ResourceRecordSets, rrset dnsprovider.ResourceRecordSet) {
	var found dnsprovider.ResourceRecordSet
	for _, r := range listRrsOrFail(t, rrsets) {
		if r.Name() != rrset.Name() || r.Type() != rrset.Type() {
			continue
		}

		if found != nil {
			t.Errorf("found duplicate resource record set: %q and %q", r, found)
		}
		found = r
	}
	if found == nil {
		t.Errorf("resource record set %v not found", rrset)
	} else {
		assertEquivalent(t, found, rrset)
	}
}

// assertNotHasRecord tests that rrsets does not have a record matching name and type
func assertNotHasRecord(t *testing.T, rrsets dnsprovider.ResourceRecordSets, name string, rrstype rrstype.RrsType) {
	var found dnsprovider.ResourceRecordSet
	for _, r := range listRrsOrFail(t, rrsets) {
		if r.Name() != name || r.Type() != rrstype {
			continue
		}

		if found != nil {
			t.Errorf("found duplicate resource record set: %q and %q", r, found)
		}
		found = r
	}
	if found != nil {
		t.Errorf("resource record set found unexpectedly: %v", found)
	}
}

// assertEquivalent tests that l is equal to r, for the methods in ResourceRecordSet
func assertEquivalent(t *testing.T, l, r dnsprovider.ResourceRecordSet) {
	if l.Name() != r.Name() {
		t.Errorf("resource record sets not equal %v vs %v", l, r)
	}
	if l.Type() != r.Type() {
		t.Errorf("resource record sets not equal %v vs %v", l, r)
	}
	if l.Ttl() != r.Ttl() {
		t.Errorf("resource record sets not equal %v vs %v", l, r)
	}
	if !reflect.DeepEqual(l.Rrdatas(), r.Rrdatas()) {
		t.Errorf("resource record sets not equal %v vs %v", l, r)
	}
}

func addRrsetOrFail(t *testing.T, rrsets dnsprovider.ResourceRecordSets, rrset dnsprovider.ResourceRecordSet) {
	err := rrsets.StartChangeset().Add(rrset).Apply()
	if err != nil {
		t.Fatalf("Failed to add recordsets: %v", err)
	}
}
