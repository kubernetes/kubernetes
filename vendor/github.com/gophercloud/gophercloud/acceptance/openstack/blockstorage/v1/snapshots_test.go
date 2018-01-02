// +build acceptance blockstorage

package v1

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/blockstorage/v1/snapshots"
)

func TestSnapshotsList(t *testing.T) {
	client, err := clients.NewBlockStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	allPages, err := snapshots.List(client, snapshots.ListOpts{}).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve snapshots: %v", err)
	}

	allSnapshots, err := snapshots.ExtractSnapshots(allPages)
	if err != nil {
		t.Fatalf("Unable to extract snapshots: %v", err)
	}

	for _, snapshot := range allSnapshots {
		tools.PrintResource(t, snapshot)
	}
}

func TestSnapshotsCreateDelete(t *testing.T) {
	client, err := clients.NewBlockStorageV1Client()
	if err != nil {
		t.Fatalf("Unable to create a blockstorage client: %v", err)
	}

	volume, err := CreateVolume(t, client)
	if err != nil {
		t.Fatalf("Unable to create volume: %v", err)
	}
	defer DeleteVolume(t, client, volume)

	snapshot, err := CreateSnapshot(t, client, volume)
	if err != nil {
		t.Fatalf("Unable to create snapshot: %v", err)
	}
	defer DeleteSnapshotshot(t, client, snapshot)

	newSnapshot, err := snapshots.Get(client, snapshot.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve snapshot: %v", err)
	}

	tools.PrintResource(t, newSnapshot)
}
