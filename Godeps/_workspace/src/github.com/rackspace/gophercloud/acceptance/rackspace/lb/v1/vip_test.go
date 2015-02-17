// +build acceptance lbs

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/lbs"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/vips"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestVIPs(t *testing.T) {
	client := setup(t)

	ids := createLB(t, client, 1)
	lbID := ids[0]

	listVIPs(t, client, lbID)

	vipIDs := addVIPs(t, client, lbID, 3)

	deleteVIP(t, client, lbID, vipIDs[0])

	bulkDeleteVIPs(t, client, lbID, vipIDs[1:])

	waitForLB(client, lbID, lbs.ACTIVE)
	deleteLB(t, client, lbID)
}

func listVIPs(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	err := vips.List(client, lbID).EachPage(func(page pagination.Page) (bool, error) {
		vipList, err := vips.ExtractVIPs(page)
		th.AssertNoErr(t, err)

		for _, vip := range vipList {
			t.Logf("Listing VIP: ID [%s] Address [%s] Type [%s] Version [%s]",
				vip.ID, vip.Address, vip.Type, vip.Version)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)
}

func addVIPs(t *testing.T, client *gophercloud.ServiceClient, lbID, count int) []int {
	ids := []int{}

	for i := 0; i < count; i++ {
		opts := vips.CreateOpts{
			Type:    vips.PUBLIC,
			Version: vips.IPV6,
		}

		vip, err := vips.Create(client, lbID, opts).Extract()
		th.AssertNoErr(t, err)

		t.Logf("Created VIP %d", vip.ID)

		waitForLB(client, lbID, lbs.ACTIVE)

		ids = append(ids, vip.ID)
	}

	return ids
}

func deleteVIP(t *testing.T, client *gophercloud.ServiceClient, lbID, vipID int) {
	err := vips.Delete(client, lbID, vipID).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Deleted VIP %d", vipID)

	waitForLB(client, lbID, lbs.ACTIVE)
}

func bulkDeleteVIPs(t *testing.T, client *gophercloud.ServiceClient, lbID int, ids []int) {
	err := vips.BulkDelete(client, lbID, ids).ExtractErr()
	th.AssertNoErr(t, err)
	t.Logf("Deleted VIPs %s", intsToStr(ids))
}
