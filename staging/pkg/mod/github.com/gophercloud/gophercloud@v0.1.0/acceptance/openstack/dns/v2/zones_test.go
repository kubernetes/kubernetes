// +build acceptance dns zones

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/dns/v2/zones"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestZonesCRUD(t *testing.T) {
	clients.RequireDNS(t)

	client, err := clients.NewDNSV2Client()
	th.AssertNoErr(t, err)

	zone, err := CreateZone(t, client)
	th.AssertNoErr(t, err)
	defer DeleteZone(t, client, zone)

	tools.PrintResource(t, &zone)

	allPages, err := zones.List(client, nil).AllPages()
	th.AssertNoErr(t, err)

	allZones, err := zones.ExtractZones(allPages)
	th.AssertNoErr(t, err)

	var found bool
	for _, z := range allZones {
		tools.PrintResource(t, &z)

		if zone.Name == z.Name {
			found = true
		}
	}

	th.AssertEquals(t, found, true)

	description := ""
	updateOpts := zones.UpdateOpts{
		Description: &description,
		TTL:         0,
	}

	newZone, err := zones.Update(client, zone.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, &newZone)

	th.AssertEquals(t, newZone.Description, description)
}
