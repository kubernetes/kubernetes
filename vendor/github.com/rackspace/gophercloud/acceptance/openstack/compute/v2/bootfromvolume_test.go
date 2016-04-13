// +build acceptance

package v2

import (
	"testing"

	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack/compute/v2/extensions/bootfromvolume"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	th "github.com/rackspace/gophercloud/testhelper"
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
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
	}
	server, err := bootfromvolume.Create(client, bootfromvolume.CreateOptsExt{
		serverCreateOpts,
		bd,
	}).Extract()
	th.AssertNoErr(t, err)
	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}

	t.Logf("Created server: %+v\n", server)
	defer servers.Delete(client, server.ID)
	t.Logf("Deleting server [%s]...", name)
}

func TestMultiEphemeral(t *testing.T) {
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
			BootIndex:           0,
			UUID:                choices.ImageID,
			SourceType:          bootfromvolume.Image,
			DestinationType:     "local",
			DeleteOnTermination: true,
		},
		bootfromvolume.BlockDevice{
			BootIndex:           -1,
			SourceType:          bootfromvolume.Blank,
			DestinationType:     "local",
			DeleteOnTermination: true,
			GuestFormat:         "ext4",
			VolumeSize:          1,
		},
		bootfromvolume.BlockDevice{
			BootIndex:           -1,
			SourceType:          bootfromvolume.Blank,
			DestinationType:     "local",
			DeleteOnTermination: true,
			GuestFormat:         "ext4",
			VolumeSize:          1,
		},
	}

	serverCreateOpts := servers.CreateOpts{
		Name:      name,
		FlavorRef: choices.FlavorID,
		ImageRef:  choices.ImageID,
	}
	server, err := bootfromvolume.Create(client, bootfromvolume.CreateOptsExt{
		serverCreateOpts,
		bd,
	}).Extract()
	th.AssertNoErr(t, err)
	if err = waitForStatus(client, server, "ACTIVE"); err != nil {
		t.Fatal(err)
	}

	t.Logf("Created server: %+v\n", server)
	defer servers.Delete(client, server.ID)
	t.Logf("Deleting server [%s]...", name)
}
