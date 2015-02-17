package acl

import (
	"testing"

	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

const (
	lbID    = 12345
	itemID1 = 67890
	itemID2 = 67891
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockListResponse(t, lbID)

	count := 0

	err := List(client.ServiceClient(), lbID).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := ExtractAccessList(page)
		th.AssertNoErr(t, err)

		expected := AccessList{
			NetworkItem{Address: "206.160.163.21", ID: 21, Type: DENY},
			NetworkItem{Address: "206.160.163.22", ID: 22, Type: DENY},
			NetworkItem{Address: "206.160.163.23", ID: 23, Type: DENY},
			NetworkItem{Address: "206.160.163.24", ID: 24, Type: DENY},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, count)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockCreateResponse(t, lbID)

	opts := CreateOpts{
		CreateOpt{Address: "206.160.163.21", Type: DENY},
		CreateOpt{Address: "206.160.165.11", Type: DENY},
	}

	err := Create(client.ServiceClient(), lbID, opts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestBulkDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	ids := []int{itemID1, itemID2}

	mockBatchDeleteResponse(t, lbID, ids)

	err := BulkDelete(client.ServiceClient(), lbID, ids).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteResponse(t, lbID, itemID1)

	err := Delete(client.ServiceClient(), lbID, itemID1).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestDeleteAll(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	mockDeleteAllResponse(t, lbID)

	err := DeleteAll(client.ServiceClient(), lbID).ExtractErr()
	th.AssertNoErr(t, err)
}
