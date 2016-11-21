// +build acceptance blockstorage

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/volumetypes"
)

func TestVolumeTypesList(t *testing.T) {
	client, err := clients.NewBlockStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	allPages, err := volumetypes.List(client).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve volume types: %v", err)
	}

	allVolumeTypes, err := volumetypes.ExtractVolumeTypes(allPages)
	if err != nil {
		t.Fatalf("Unable to extract volume types: %v", err)
	}

	for _, volumeType := range allVolumeTypes {
		PrintVolumeType(t, &volumeType)
	}
}

func TestVolumeTypesCreateDestroy(t *testing.T) {
	client, err := clients.NewBlockStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	volumeType, err := CreateVolumeType(t, client)
	if err != nil {
		t.Fatalf("Unable to create volume type: %v", err)
	}
	defer DeleteVolumeType(t, client, volumeType)

	PrintVolumeType(t, volumeType)
}
