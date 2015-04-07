// +build acceptance compute servers

package v2

import (
	"os"
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack"
	osVolumes "github.com/rackspace/gophercloud/openstack/blockstorage/v1/volumes"
	osVolumeAttach "github.com/rackspace/gophercloud/openstack/compute/v2/extensions/volumeattach"
	osServers "github.com/rackspace/gophercloud/openstack/compute/v2/servers"
	"github.com/rackspace/gophercloud/rackspace"
	"github.com/rackspace/gophercloud/rackspace/blockstorage/v1/volumes"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/servers"
	"github.com/rackspace/gophercloud/rackspace/compute/v2/volumeattach"
	th "github.com/rackspace/gophercloud/testhelper"
)

func newBlockClient(t *testing.T) (*gophercloud.ServiceClient, error) {
	ao, err := rackspace.AuthOptionsFromEnv()
	th.AssertNoErr(t, err)

	client, err := rackspace.AuthenticatedClient(ao)
	th.AssertNoErr(t, err)

	return openstack.NewBlockStorageV1(client, gophercloud.EndpointOpts{
		Region: os.Getenv("RS_REGION_NAME"),
	})
}

func createVAServer(t *testing.T, computeClient *gophercloud.ServiceClient, choices *serverOpts) (*osServers.Server, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	name := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create server: %s\n", name)

	pwd := tools.MakeNewPassword("")

	server, err := servers.Create(computeClient, osServers.CreateOpts{
		Name:      name,
		FlavorRef: choices.flavorID,
		ImageRef:  choices.imageID,
		AdminPass: pwd,
	}).Extract()
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}

	th.AssertEquals(t, pwd, server.AdminPass)

	return server, err
}

func createVAVolume(t *testing.T, blockClient *gophercloud.ServiceClient) (*volumes.Volume, error) {
	volume, err := volumes.Create(blockClient, &osVolumes.CreateOpts{
		Size: 80,
		Name: "gophercloud-test-volume",
	}).Extract()
	th.AssertNoErr(t, err)
	defer func() {
		err = osVolumes.WaitForStatus(blockClient, volume.ID, "available", 60)
		th.AssertNoErr(t, err)
	}()

	return volume, err
}

func createVolumeAttachment(t *testing.T, computeClient *gophercloud.ServiceClient, blockClient *gophercloud.ServiceClient, serverID string, volumeID string) {
	va, err := volumeattach.Create(computeClient, serverID, &osVolumeAttach.CreateOpts{
		VolumeID: volumeID,
	}).Extract()
	th.AssertNoErr(t, err)
	defer func() {
		err = osVolumes.WaitForStatus(blockClient, volumeID, "in-use", 60)
		th.AssertNoErr(t, err)
		err = volumeattach.Delete(computeClient, serverID, va.ID).ExtractErr()
		th.AssertNoErr(t, err)
		err = osVolumes.WaitForStatus(blockClient, volumeID, "available", 60)
		th.AssertNoErr(t, err)
	}()
	t.Logf("Attached volume to server: %+v", va)
}

func TestAttachVolume(t *testing.T) {
	choices, err := optionsFromEnv()
	if err != nil {
		t.Fatal(err)
	}

	computeClient, err := newClient()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	blockClient, err := newBlockClient(t)
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	server, err := createVAServer(t, computeClient, choices)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer func() {
		servers.Delete(computeClient, server.ID)
		t.Logf("Server deleted.")
	}()

	if err = osServers.WaitForStatus(computeClient, server.ID, "ACTIVE", 300); err != nil {
		t.Fatalf("Unable to wait for server: %v", err)
	}

	volume, err := createVAVolume(t, blockClient)
	if err != nil {
		t.Fatalf("Unable to create volume: %v", err)
	}
	defer func() {
		err = volumes.Delete(blockClient, volume.ID).ExtractErr()
		th.AssertNoErr(t, err)
		t.Logf("Volume deleted.")
	}()

	createVolumeAttachment(t, computeClient, blockClient, server.ID, volume.ID)

}
