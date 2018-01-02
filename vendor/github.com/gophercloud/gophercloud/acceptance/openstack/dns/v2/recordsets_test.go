// +build acceptance dns recordsets

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/dns/v2/recordsets"
)

func TestRecordSetsListByZone(t *testing.T) {
	client, err := clients.NewDNSV2Client()
	if err != nil {
		t.Fatalf("Unable to create a DNS client: %v", err)
	}

	zone, err := CreateZone(t, client)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteZone(t, client, zone)

	var allRecordSets []recordsets.RecordSet
	allPages, err := recordsets.ListByZone(client, zone.ID, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve recordsets: %v", err)
	}

	allRecordSets, err = recordsets.ExtractRecordSets(allPages)
	if err != nil {
		t.Fatalf("Unable to extract recordsets: %v", err)
	}

	for _, recordset := range allRecordSets {
		tools.PrintResource(t, &recordset)
	}
}

func TestRecordSetsListByZoneLimited(t *testing.T) {
	client, err := clients.NewDNSV2Client()
	if err != nil {
		t.Fatalf("Unable to create a DNS client: %v", err)
	}

	zone, err := CreateZone(t, client)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteZone(t, client, zone)

	var allRecordSets []recordsets.RecordSet
	listOpts := recordsets.ListOpts{
		Limit: 1,
	}
	allPages, err := recordsets.ListByZone(client, zone.ID, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve recordsets: %v", err)
	}

	allRecordSets, err = recordsets.ExtractRecordSets(allPages)
	if err != nil {
		t.Fatalf("Unable to extract recordsets: %v", err)
	}

	for _, recordset := range allRecordSets {
		tools.PrintResource(t, &recordset)
	}
}

func TestRecordSetCRUD(t *testing.T) {
	client, err := clients.NewDNSV2Client()
	if err != nil {
		t.Fatalf("Unable to create a DNS client: %v", err)
	}

	zone, err := CreateZone(t, client)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteZone(t, client, zone)

	tools.PrintResource(t, &zone)

	rs, err := CreateRecordSet(t, client, zone)
	if err != nil {
		t.Fatal(err)
	}
	defer DeleteRecordSet(t, client, rs)

	tools.PrintResource(t, &rs)

	updateOpts := recordsets.UpdateOpts{
		Description: "New description",
		TTL:         0,
	}

	newRS, err := recordsets.Update(client, rs.ZoneID, rs.ID, updateOpts).Extract()
	if err != nil {
		t.Fatal(err)
	}

	tools.PrintResource(t, &newRS)
}
