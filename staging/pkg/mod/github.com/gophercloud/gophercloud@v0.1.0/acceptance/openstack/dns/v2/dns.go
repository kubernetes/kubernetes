package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/dns/v2/recordsets"
	"github.com/gophercloud/gophercloud/openstack/dns/v2/zones"
	th "github.com/gophercloud/gophercloud/testhelper"
)

// CreateRecordSet will create a RecordSet with a random name. An error will
// be returned if the zone was unable to be created.
func CreateRecordSet(t *testing.T, client *gophercloud.ServiceClient, zone *zones.Zone) (*recordsets.RecordSet, error) {
	t.Logf("Attempting to create recordset: %s", zone.Name)

	createOpts := recordsets.CreateOpts{
		Name:        zone.Name,
		Type:        "A",
		TTL:         3600,
		Description: "Test recordset",
		Records:     []string{"10.1.0.2"},
	}

	rs, err := recordsets.Create(client, zone.ID, createOpts).Extract()
	if err != nil {
		return rs, err
	}

	if err := WaitForRecordSetStatus(client, rs, "ACTIVE"); err != nil {
		return rs, err
	}

	newRS, err := recordsets.Get(client, rs.ZoneID, rs.ID).Extract()
	if err != nil {
		return newRS, err
	}

	t.Logf("Created record set: %s", newRS.Name)

	th.AssertEquals(t, newRS.Name, zone.Name)

	return rs, nil
}

// CreateZone will create a Zone with a random name. An error will
// be returned if the zone was unable to be created.
func CreateZone(t *testing.T, client *gophercloud.ServiceClient) (*zones.Zone, error) {
	zoneName := tools.RandomString("ACPTTEST", 8) + ".com."

	t.Logf("Attempting to create zone: %s", zoneName)
	createOpts := zones.CreateOpts{
		Name:        zoneName,
		Email:       "root@example.com",
		Type:        "PRIMARY",
		TTL:         7200,
		Description: "Test zone",
	}

	zone, err := zones.Create(client, createOpts).Extract()
	if err != nil {
		return zone, err
	}

	if err := WaitForZoneStatus(client, zone, "ACTIVE"); err != nil {
		return zone, err
	}

	newZone, err := zones.Get(client, zone.ID).Extract()
	if err != nil {
		return zone, err
	}

	t.Logf("Created Zone: %s", zoneName)

	th.AssertEquals(t, newZone.Name, zoneName)
	th.AssertEquals(t, newZone.TTL, 7200)

	return newZone, nil
}

// CreateSecondaryZone will create a Zone with a random name. An error will
// be returned if the zone was unable to be created.
//
// This is only for example purposes as it will try to do a zone transfer.
func CreateSecondaryZone(t *testing.T, client *gophercloud.ServiceClient) (*zones.Zone, error) {
	zoneName := tools.RandomString("ACPTTEST", 8) + ".com."

	t.Logf("Attempting to create zone: %s", zoneName)
	createOpts := zones.CreateOpts{
		Name:    zoneName,
		Type:    "SECONDARY",
		Masters: []string{"10.0.0.1"},
	}

	zone, err := zones.Create(client, createOpts).Extract()
	if err != nil {
		return zone, err
	}

	if err := WaitForZoneStatus(client, zone, "ACTIVE"); err != nil {
		return zone, err
	}

	newZone, err := zones.Get(client, zone.ID).Extract()
	if err != nil {
		return zone, err
	}

	t.Logf("Created Zone: %s", zoneName)

	th.AssertEquals(t, newZone.Name, zoneName)
	th.AssertEquals(t, newZone.Masters[0], "10.0.0.1")

	return newZone, nil
}

// DeleteRecordSet will delete a specified record set. A fatal error will occur if
// the record set failed to be deleted. This works best when used as a deferred
// function.
func DeleteRecordSet(t *testing.T, client *gophercloud.ServiceClient, rs *recordsets.RecordSet) {
	err := recordsets.Delete(client, rs.ZoneID, rs.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete record set %s: %v", rs.ID, err)
	}

	t.Logf("Deleted record set: %s", rs.ID)
}

// DeleteZone will delete a specified zone. A fatal error will occur if
// the zone failed to be deleted. This works best when used as a deferred
// function.
func DeleteZone(t *testing.T, client *gophercloud.ServiceClient, zone *zones.Zone) {
	_, err := zones.Delete(client, zone.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to delete zone %s: %v", zone.ID, err)
	}

	t.Logf("Deleted zone: %s", zone.ID)
}

// WaitForRecordSetStatus will poll a record set's status until it either matches
// the specified status or the status becomes ERROR.
func WaitForRecordSetStatus(client *gophercloud.ServiceClient, rs *recordsets.RecordSet, status string) error {
	return gophercloud.WaitFor(600, func() (bool, error) {
		current, err := recordsets.Get(client, rs.ZoneID, rs.ID).Extract()
		if err != nil {
			return false, err
		}

		if current.Status == status {
			return true, nil
		}

		return false, nil
	})
}

// WaitForZoneStatus will poll a zone's status until it either matches
// the specified status or the status becomes ERROR.
func WaitForZoneStatus(client *gophercloud.ServiceClient, zone *zones.Zone, status string) error {
	return gophercloud.WaitFor(600, func() (bool, error) {
		current, err := zones.Get(client, zone.ID).Extract()
		if err != nil {
			return false, err
		}

		if current.Status == status {
			return true, nil
		}

		return false, nil
	})
}
