// +build acceptance blockstorage

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumes"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestVolumesList(t *testing.T) {
	client, err := clients.NewBlockStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	allPages, err := volumes.List(client, volumes.ListOpts{}).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve volumes: %v", err)
	}

	allVolumes, err := volumes.ExtractVolumes(allPages)
	if err != nil {
		t.Fatalf("Unable to extract volumes: %v", err)
	}

	for _, volume := range allVolumes {
		tools.PrintResource(t, volume)
	}
}

func TestVolumesCreateDestroy(t *testing.T) {
	client, err := clients.NewBlockStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create blockstorage client: %v", err)
	}

	volume, err := CreateVolume(t, client)
	if err != nil {
		t.Fatalf("Unable to create volume: %v", err)
	}
	defer DeleteVolume(t, client, volume)

	newVolume, err := volumes.Get(client, volume.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve volume: %v", err)
	}

	tools.PrintResource(t, newVolume)
	th.AssertEquals(t, volume.Name, newVolume.Name)
	th.AssertEquals(t, volume.Description, newVolume.Description)

	// Update volume
	updatedVolumeName := ""
	updatedVolumeDescription := ""
	updateOpts := volumes.UpdateOpts{
		Name:        &updatedVolumeName,
		Description: &updatedVolumeDescription,
	}
	updatedVolume, err := volumes.Update(client, volume.ID, updateOpts).Extract()
	th.AssertNoErr(t, err)

	tools.PrintResource(t, updatedVolume)
	th.AssertEquals(t, updatedVolume.Name, updatedVolumeName)
	th.AssertEquals(t, updatedVolume.Description, updatedVolumeDescription)
}
