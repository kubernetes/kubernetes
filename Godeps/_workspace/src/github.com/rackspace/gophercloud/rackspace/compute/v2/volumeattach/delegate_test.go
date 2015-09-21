package volumeattach

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleListSuccessfully(t)
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"

	count := 0
	err := List(client.ServiceClient(), serverId).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := os.ExtractVolumeAttachments(page)
		th.AssertNoErr(t, err)
		th.CheckDeepEquals(t, os.ExpectedVolumeAttachmentSlice, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, 1, count)
}

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleCreateSuccessfully(t)
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"

	actual, err := Create(client.ServiceClient(), serverId, os.CreateOpts{
		Device:   "/dev/vdc",
		VolumeID: "a26887c6-c47b-4654-abb5-dfadf7d3f804",
	}).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &os.CreatedVolumeAttachment, actual)
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleGetSuccessfully(t)
	aId := "a26887c6-c47b-4654-abb5-dfadf7d3f804"
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"

	actual, err := Get(client.ServiceClient(), serverId, aId).Extract()
	th.AssertNoErr(t, err)
	th.CheckDeepEquals(t, &os.SecondVolumeAttachment, actual)
}

func TestDelete(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleDeleteSuccessfully(t)
	aId := "a26887c6-c47b-4654-abb5-dfadf7d3f804"
	serverId := "4d8c3732-a248-40ed-bebc-539a6ffd25c0"

	err := Delete(client.ServiceClient(), serverId, aId).ExtractErr()
	th.AssertNoErr(t, err)
}
