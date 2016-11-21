// Package v2 contains common functions for creating block storage based
// resources for use in acceptance tests. See the `*_test.go` files for
// example usages.
package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v2/volumes"
)

// CreateVolume will create a volume with a random name and size of 1GB. An
// error will be returned if the volume was unable to be created.
func CreateVolume(t *testing.T, client *gophercloud.ServiceClient) (*volumes.Volume, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires volume creation in short mode.")
	}

	volumeName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create volume: %s", volumeName)

	createOpts := volumes.CreateOpts{
		Size: 1,
		Name: volumeName,
	}

	volume, err := volumes.Create(client, createOpts).Extract()
	if err != nil {
		return volume, err
	}

	err = volumes.WaitForStatus(client, volume.ID, "available", 60)
	if err != nil {
		return volume, err
	}

	return volume, nil
}

// CreateVolumeFromImage will create a volume from with a random name and size of
// 1GB. An error will be returned if the volume was unable to be created.
func CreateVolumeFromImage(t *testing.T, client *gophercloud.ServiceClient, choices *clients.AcceptanceTestChoices) (*volumes.Volume, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires volume creation in short mode.")
	}

	volumeName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create volume: %s", volumeName)

	createOpts := volumes.CreateOpts{
		Size:    1,
		Name:    volumeName,
		ImageID: choices.ImageID,
	}

	volume, err := volumes.Create(client, createOpts).Extract()
	if err != nil {
		return volume, err
	}

	err = volumes.WaitForStatus(client, volume.ID, "available", 60)
	if err != nil {
		return volume, err
	}

	return volume, nil
}

// DeleteVolume will delete a volume. A fatal error will occur if the volume
// failed to be deleted. This works best when used as a deferred function.
func DeleteVolume(t *testing.T, client *gophercloud.ServiceClient, volume *volumes.Volume) {
	err := volumes.Delete(client, volume.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Unable to delete volume %s: %v", volume.ID, err)
	}

	t.Logf("Deleted volume: %s", volume.ID)
}

// PrintVolume will print a volume and all of its attributes.
func PrintVolume(t *testing.T, volume *volumes.Volume) {
	t.Logf("ID: %s", volume.ID)
	t.Logf("Status: %s", volume.Status)
	t.Logf("Size: %d", volume.Size)
	t.Logf("AvailabilityZone: %s", volume.AvailabilityZone)
	t.Logf("CreatedAt: %v", volume.CreatedAt)
	t.Logf("UpdatedAt: %v", volume.CreatedAt)
	t.Logf("Attachments: %#v", volume.Attachments)
	t.Logf("Name: %s", volume.Name)
	t.Logf("Description: %s", volume.Description)
	t.Logf("VolumeType: %s", volume.VolumeType)
	t.Logf("SnapshotID: %s", volume.SnapshotID)
	t.Logf("SourceVolID: %s", volume.SourceVolID)
	t.Logf("Metadata: %#v", volume.Metadata)
	t.Logf("UserID: %s", volume.UserID)
	t.Logf("Bootable: %s", volume.Bootable)
	t.Logf("Encrypted: %s", volume.Encrypted)
	t.Logf("ReplicationStatus: %s", volume.ReplicationStatus)
	t.Logf("ConsistencyGroupID: %s", volume.ConsistencyGroupID)
	t.Logf("Multiattach: %t", volume.Multiattach)
}
