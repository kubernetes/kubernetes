// +build acceptance blockstorage volumes

package v1

import (
	"testing"

	"github.com/rackspace/gophercloud"
	os "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes"
	th "github.com/rackspace/gophercloud/testhelper"
)

func TestVolumes(t *testing.T) {
	client := setup(t)

	t.Logf("Listing volumes")
	testVolumeList(t, client)

	t.Logf("Creating volume")
	volumeID := testVolumeCreate(t, client)

	t.Logf("Getting volume %s", volumeID)
	testVolumeGet(t, client, volumeID)

	t.Logf("Updating volume %s", volumeID)
	testVolumeUpdate(t, client, volumeID)

	t.Logf("Deleting volume %s", volumeID)
	testVolumeDelete(t, client, volumeID)
}

func testVolumeList(t *testing.T, client *gophercloud.ServiceClient) {
	volumes.List(client).EachPage(func(page pagination.Page) (bool, error) {
		vList, err := volumes.ExtractVolumes(page)
		th.AssertNoErr(t, err)

		for _, v := range vList {
			t.Logf("Volume: ID [%s] Name [%s] Type [%s] Created [%s]", v.ID, v.Name,
				v.VolumeType, v.CreatedAt)
		}

		return true, nil
	})
}

func testVolumeCreate(t *testing.T, client *gophercloud.ServiceClient) string {
	vol, err := volumes.Create(client, os.CreateOpts{Size: 75}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created volume: ID [%s] Size [%s]", vol.ID, vol.Size)
	return vol.ID
}

func testVolumeGet(t *testing.T, client *gophercloud.ServiceClient, id string) {
	vol, err := volumes.Get(client, id).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created volume: ID [%s] Size [%s]", vol.ID, vol.Size)
}

func testVolumeUpdate(t *testing.T, client *gophercloud.ServiceClient, id string) {
	vol, err := volumes.Update(client, id, volumes.UpdateOpts{Name: "new_name"}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created volume: ID [%s] Name [%s]", vol.ID, vol.Name)
}

func testVolumeDelete(t *testing.T, client *gophercloud.ServiceClient, id string) {
	res := volumes.Delete(client, id)
	th.AssertNoErr(t, res.Err)
	t.Logf("Deleted volume %s", id)
}
