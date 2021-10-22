// +build acceptance dns recordsets

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/dns/v2/recordsets"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestRecordSetsListByZone(t *testing.T) {
	clients.RequireDNS(t)

	client, err := clients.NewDNSV2Client()
	th.AssertNoErr(t, err)

	zone, err := CreateZone(t, client)
	th.AssertNoErr(t, err)
	defer DeleteZone(t, client, zone)

	allPages, err := recordsets.ListByZone(client, zone.ID, nil).AllPages()
	th.AssertNoErr(t, err)

	allRecordSets, err := recordsets.ExtractRecordSets(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, recordset := range allRecordSets {
		tools.PrintResource(t, &recordset)

		if recordset.ZoneID == zone.ID {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	listOpts := recordsets.ListOpts{
		Limit: 1,
	}

	err = recordsets.ListByZone(client, zone.ID, listOpts).EachPage(
		func(page pagination.Page) (bool, error) {
			rr, err := recordsets.ExtractRecordSets(page)
			th.AssertNoErr(t, err)
			th.AssertEquals(t, len(rr), 1)
			return true, nil
		},
	)
	th.AssertNoErr(t, err)
}

func TestRecordSetsCRUD(t *testing.T) {
	clients.RequireDNS(t)

	client, err := clients.NewDNSV2Client()
	th.AssertNoErr(t, err)

	zone, err := CreateZone(t, client)
	th.AssertNoErr(t, err)
	defer DeleteZone(t, client, zone)

	tools.PrintResource(t, &zone)

	rs, err := CreateRecordSet(t, client, zone)
	th.AssertNoErr(t, err)
	defer DeleteRecordSet(t, client, rs)

	tools.PrintResource(t, &rs)

	description := ""
	updateOpts := recordsets.UpdateOpts{
		Description: &description,
	}

	newRS, err := recordsets.Update(client, rs.ZoneID, rs.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, &newRS)

	th.AssertEquals(t, newRS.Description, description)

	records := []string{"10.1.0.3"}
	updateOpts = recordsets.UpdateOpts{
		Records: records,
	}

	newRS, err = recordsets.Update(client, rs.ZoneID, rs.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, &newRS)

	th.AssertDeepEquals(t, newRS.Records, records)
	th.AssertEquals(t, newRS.TTL, 3600)

	ttl := 0
	updateOpts = recordsets.UpdateOpts{
		TTL: &ttl,
	}

	newRS, err = recordsets.Update(client, rs.ZoneID, rs.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, &newRS)

	th.AssertDeepEquals(t, newRS.Records, records)
	th.AssertEquals(t, newRS.TTL, ttl)
}
