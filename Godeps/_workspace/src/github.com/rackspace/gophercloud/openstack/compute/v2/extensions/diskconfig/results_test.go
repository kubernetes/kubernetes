package diskconfig

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestExtractGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	servers.HandleServerGetSuccessfully(t)

	config, err := ExtractGet(servers.Get(client.ServiceClient(), "1234asdf"))
	th.AssertNoErr(t, err)
	th.CheckEquals(t, Manual, *config)
}

func TestExtractUpdate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	servers.HandleServerUpdateSuccessfully(t)

	r := servers.Update(client.ServiceClient(), "1234asdf", servers.UpdateOpts{
		Name: "new-name",
	})
	config, err := ExtractUpdate(r)
	th.AssertNoErr(t, err)
	th.CheckEquals(t, Manual, *config)
}

func TestExtractRebuild(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	servers.HandleRebuildSuccessfully(t, servers.SingleServerBody)

	r := servers.Rebuild(client.ServiceClient(), "1234asdf", servers.RebuildOpts{
		Name:       "new-name",
		AdminPass:  "swordfish",
		ImageID:    "http://104.130.131.164:8774/fcad67a6189847c4aecfa3c81a05783b/images/f90f6034-2570-4974-8351-6b49732ef2eb",
		AccessIPv4: "1.2.3.4",
	})
	config, err := ExtractRebuild(r)
	th.AssertNoErr(t, err)
	th.CheckEquals(t, Manual, *config)
}

func TestExtractList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	servers.HandleServerListSuccessfully(t)

	pages := 0
	err := servers.List(client.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		config, err := ExtractDiskConfig(page, 0)
		th.AssertNoErr(t, err)
		th.CheckEquals(t, Manual, *config)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, pages, 1)
}
