// +build acceptance lbs

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/acl"
	"github.com/rackspace/gophercloud/rackspace/lb/v1/lbs"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestACL(t *testing.T) {
	client := setup(t)

	ids := createLB(t, client, 1)
	lbID := ids[0]

	createACL(t, client, lbID)

	waitForLB(client, lbID, lbs.ACTIVE)

	networkIDs := showACL(t, client, lbID)

	deleteNetworkItem(t, client, lbID, networkIDs[0])
	waitForLB(client, lbID, lbs.ACTIVE)

	bulkDeleteACL(t, client, lbID, networkIDs[1:2])
	waitForLB(client, lbID, lbs.ACTIVE)

	deleteACL(t, client, lbID)
	waitForLB(client, lbID, lbs.ACTIVE)

	deleteLB(t, client, lbID)
}

func createACL(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	opts := acl.CreateOpts{
		acl.CreateOpt{Address: "206.160.163.21", Type: acl.DENY},
		acl.CreateOpt{Address: "206.160.165.11", Type: acl.DENY},
		acl.CreateOpt{Address: "206.160.165.12", Type: acl.DENY},
		acl.CreateOpt{Address: "206.160.165.13", Type: acl.ALLOW},
	}

	err := acl.Create(client, lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)

	t.Logf("Created ACL items for LB %d", lbID)
}

func showACL(t *testing.T, client *gophercloud.ServiceClient, lbID int) []int {
	ids := []int{}

	err := acl.List(client, lbID).EachPage(func(page pagination.Page) (bool, error) {
		accessList, err := acl.ExtractAccessList(page)
		th.AssertNoErr(t, err)

		for _, i := range accessList {
			t.Logf("Listing network item: ID [%s] Address [%s] Type [%s]", i.ID, i.Address, i.Type)
			ids = append(ids, i.ID)
		}

		return true, nil
	})
	th.AssertNoErr(t, err)

	return ids
}

func deleteNetworkItem(t *testing.T, client *gophercloud.ServiceClient, lbID, itemID int) {
	err := acl.Delete(client, lbID, itemID).ExtractErr()

	th.AssertNoErr(t, err)

	t.Logf("Deleted network item %d", itemID)
}

func bulkDeleteACL(t *testing.T, client *gophercloud.ServiceClient, lbID int, items []int) {
	err := acl.BulkDelete(client, lbID, items).ExtractErr()

	th.AssertNoErr(t, err)

	t.Logf("Deleted network items %s", intsToStr(items))
}

func deleteACL(t *testing.T, client *gophercloud.ServiceClient, lbID int) {
	err := acl.DeleteAll(client, lbID).ExtractErr()

	th.AssertNoErr(t, err)

	t.Logf("Deleted ACL from LB %d", lbID)
}
