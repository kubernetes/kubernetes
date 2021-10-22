package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/shares"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestShareCreate(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create a share: %v", err)
	}

	defer DeleteShare(t, client, share)

	created, err := shares.Get(client, share.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve share: %v", err)
	}
	tools.PrintResource(t, created)
}

func TestShareUpdate(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create shared file system client: %v", err)
	}

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create share: %v", err)
	}

	defer DeleteShare(t, client, share)

	expectedShare, err := shares.Get(client, share.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve share: %v", err)
	}

	name := "NewName"
	description := ""
	iFalse := false
	options := shares.UpdateOpts{
		DisplayName:        &name,
		DisplayDescription: &description,
		IsPublic:           &iFalse,
	}

	expectedShare.Name = name
	expectedShare.Description = description
	expectedShare.IsPublic = iFalse

	_, err = shares.Update(client, share.ID, options).Extract()
	if err != nil {
		t.Errorf("Unable to update share: %v", err)
	}

	updatedShare, err := shares.Get(client, share.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve share: %v", err)
	}

	// Update time has to be set in order to get the assert equal to pass
	expectedShare.UpdatedAt = updatedShare.UpdatedAt

	tools.PrintResource(t, share)

	th.CheckDeepEquals(t, expectedShare, updatedShare)
}

func TestShareListDetail(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create a share: %v", err)
	}

	defer DeleteShare(t, client, share)

	ss, err := ListShares(t, client)
	if err != nil {
		t.Fatalf("Unable to list shares: %v", err)
	}

	for i := range ss {
		tools.PrintResource(t, &ss[i])
	}
}

func TestGrantAndRevokeAccess(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}
	client.Microversion = "2.7"

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create a share: %v", err)
	}

	defer DeleteShare(t, client, share)

	accessRight, err := GrantAccess(t, client, share)
	if err != nil {
		t.Fatalf("Unable to grant access: %v", err)
	}

	tools.PrintResource(t, accessRight)

	if err = RevokeAccess(t, client, share, accessRight); err != nil {
		t.Fatalf("Unable to revoke access: %v", err)
	}
}

func TestListAccessRights(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}
	client.Microversion = "2.7"

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create a share: %v", err)
	}

	defer DeleteShare(t, client, share)

	_, err = GrantAccess(t, client, share)
	if err != nil {
		t.Fatalf("Unable to grant access: %v", err)
	}

	rs, err := GetAccessRightsSlice(t, client, share)
	if err != nil {
		t.Fatalf("Unable to retrieve list of access rules for share %s: %v", share.ID, err)
	}

	if len(rs) != 1 {
		t.Fatalf("Unexpected number of access rules for share %s: got %d, expected 1", share.ID, len(rs))
	}

	t.Logf("Share %s has %d access rule(s):", share.ID, len(rs))

	for _, r := range rs {
		tools.PrintResource(t, &r)
	}
}

func TestExtendAndShrink(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}
	client.Microversion = "2.7"

	share, err := CreateShare(t, client)
	if err != nil {
		t.Fatalf("Unable to create a share: %v", err)
	}

	defer DeleteShare(t, client, share)

	err = ExtendShare(t, client, share, 2)
	if err != nil {
		t.Fatalf("Unable to extend a share: %v", err)
	}

	// We need to wait till the Extend operation is done
	err = waitForStatus(t, client, share.ID, "available", 120)
	if err != nil {
		t.Fatalf("Share status error: %v", err)
	}

	t.Logf("Share %s successfuly extended", share.ID)

	/* disable shrinking for the LVM dhss=false
	err = ShrinkShare(t, client, share, 1)
	if err != nil {
		t.Fatalf("Unable to shrink a share: %v", err)
	}

	// We need to wait till the Shrink operation is done
	err = waitForStatus(t, client, share.ID, "available", 300)
	if err != nil {
		t.Fatalf("Share status error: %v", err)
	}

	t.Logf("Share %s successfuly shrunk", share.ID)
	*/
}
