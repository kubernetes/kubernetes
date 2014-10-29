// +build acceptance blockstorage volumetypes

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumetypes"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestAll(t *testing.T) {
	client := setup(t)

	t.Logf("Listing volume types")
	id := testList(t, client)

	t.Logf("Getting volume type %s", id)
	testGet(t, client, id)
}

func testList(t *testing.T, client *gophercloud.ServiceClient) string {
	var lastID string

	volumetypes.List(client).EachPage(func(page pagination.Page) (bool, error) {
		typeList, err := volumetypes.ExtractVolumeTypes(page)
		th.AssertNoErr(t, err)

		for _, vt := range typeList {
			t.Logf("Volume type: ID [%s] Name [%s]", vt.ID, vt.Name)
			lastID = vt.ID
		}

		return true, nil
	})

	return lastID
}

func testGet(t *testing.T, client *gophercloud.ServiceClient, id string) {
	vt, err := volumetypes.Get(client, id).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Volume: ID [%s] Name [%s]", vt.ID, vt.Name)
}
