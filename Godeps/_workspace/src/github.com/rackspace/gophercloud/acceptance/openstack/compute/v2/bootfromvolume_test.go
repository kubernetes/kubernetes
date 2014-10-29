// +build acceptance

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/smashwilson/gophercloud/acceptance/tools"
)

func TestBootFromVolume(t *testing.T) {
	client, err := newClient()
	th.AssertNoErr(t, err)

	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	choices, err := ComputeChoicesFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	name := tools.RandomString("Gophercloud-", 8)
	t.Logf("Creating server [%s].", name)

	bd := []bootfromvolume.BlockDevice{
		bootfromvolume.BlockDevice{
			UUID:       choices.ImageID,
			SourceType: bootfromvolume.Image,
			VolumeSize: 10,
		},
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      name,
		FlavorRef: "3",
	}
	server, err := bootfromvolume.Create(client, bootfromvolume.CreateOptsExt{
		serverCreateOpts,
		bd,
	}).Extract()
	th.AssertNoErr(t, err)
	t.Logf("Created server: %+v\n", server)
	//defer deleteServer(t, client, server)
	t.Logf("Deleting server [%s]...", name)
}
