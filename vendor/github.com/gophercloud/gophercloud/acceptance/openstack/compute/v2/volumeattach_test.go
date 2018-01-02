// +build acceptance compute volumeattach

package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes"
)

func TestVolumeAttachAttachment(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping test that requires server creation in short mode.")
	}

	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	blockClient, err := clients.NewBlockStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	server, err := CreateServer(t, client)
	if err != nil {
		t.Fatalf("Unable to create server: %v", err)
	}
	defer DeleteServer(t, client, server)

	volume, err := createVolume(t, blockClient)
	if err != nil {
		t.Fatalf("Unable to create volume: %v", err)
	}

	if err = volumes.WaitForStatus(blockClient, volume.ID, "available", 60); err != nil {
		t.Fatalf("Unable to wait for volume: %v", err)
	}
	defer deleteVolume(t, blockClient, volume)

	volumeAttachment, err := CreateVolumeAttachment(t, client, blockClient, server, volume)
	if err != nil {
		t.Fatalf("Unable to attach volume: %v", err)
	}
	defer DeleteVolumeAttachment(t, client, blockClient, server, volumeAttachment)

	tools.PrintResource(t, volumeAttachment)

}

func createVolume(t *testing.T, blockClient *gophercloud.ServiceClient) (*volumes.Volume, error) {
	volumeName := tools.RandomString("ACPTTEST", 16)
	createOpts := volumes.CreateOpts{
		Size: 1,
		Name: volumeName,
	}

	volume, err := volumes.Create(blockClient, createOpts).Extract()
	if err != nil {
		return volume, err
	}

	t.Logf("Created volume: %s", volume.ID)
	return volume, nil
}

func deleteVolume(t *testing.T, blockClient *gophercloud.ServiceClient, volume *volumes.Volume) {
	err := volumes.Delete(blockClient, volume.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete volume: %v", err)
	}

	t.Logf("Deleted volume: %s", volume.ID)
}
