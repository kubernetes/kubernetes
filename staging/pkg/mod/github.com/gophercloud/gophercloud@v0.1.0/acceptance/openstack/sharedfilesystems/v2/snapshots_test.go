package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/snapshots"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestSnapshotCreate(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create a share: %v", err)
	}

	defer DeleteShare(t, client, share)

	snapshot, err := CreateSnapshot(t, client, share.ID)
	if err != nil {
		t.Fatalf("Unable to create a snapshot: %v", err)
	}

	defer DeleteSnapshot(t, client, snapshot)

	created, err := snapshots.Get(client, snapshot.ID).Extract()
	if err != nil {
		t.Fatalf("Unable to retrieve a snapshot: %v", err)
	}

	tools.PrintResource(t, created)
}

func TestSnapshotUpdate(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create shared file system client: %v", err)
	}

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create share: %v", err)
	}

	defer DeleteShare(t, client, share)

	snapshot, err := CreateSnapshot(t, client, share.ID)
	if err != nil {
		t.Fatalf("Unable to create a snapshot: %v", err)
	}

	defer DeleteSnapshot(t, client, snapshot)

	expectedSnapshot, err := snapshots.Get(client, snapshot.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve snapshot: %v", err)
	}

	name := "NewName"
	description := ""
	options := snapshots.UpdateOpts{
		DisplayName:        &name,
		DisplayDescription: &description,
	}

	expectedSnapshot.Name = name
	expectedSnapshot.Description = description

	_, err = snapshots.Update(client, snapshot.ID, options).Extract()
	if err != nil {
		t.Errorf("Unable to update snapshot: %v", err)
	}

	updatedSnapshot, err := snapshots.Get(client, snapshot.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve snapshot: %v", err)
	}

	tools.PrintResource(t, snapshot)

	th.CheckDeepEquals(t, expectedSnapshot, updatedSnapshot)
}

func TestSnapshotListDetail(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create a share: %v", err)
	}

	defer DeleteShare(t, client, share)

	snapshot, err := CreateSnapshot(t, client, share.ID)
	if err != nil {
		t.Fatalf("Unable to create a snapshot: %v", err)
	}

	defer DeleteSnapshot(t, client, snapshot)

	ss, err := ListSnapshots(t, client)
	if err != nil {
		t.Fatalf("Unable to list snapshots: %v", err)
	}

	for i := range ss {
		tools.PrintResource(t, &ss[i])
	}
}
