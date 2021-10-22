package v2

import (
	"fmt"
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/tools"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/messages"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/shares"
)

// CreateShare will create a share with a name, and a size of 1Gb. An
// error will be returned if the share could not be created
func CreateShare(t *testing.T, client *gophercloud.ServiceClient) (*shares.Share, error) {
	if testing.Short() {
		t.Skip("Skipping test that requres share creation in short mode.")
	}

	iTrue := true
	createOpts := shares.CreateOpts{
		Size:        1,
		Name:        "My Test Share",
		Description: "My Test Description",
		ShareProto:  "NFS",
		ShareType:   "dhss_false",
		IsPublic:    &iTrue,
	}

	share, err := shares.Create(client, createOpts).Extract()
	if err != nil {
		t.Logf("Failed to create share")
		return nil, err
	}

	err = waitForStatus(t, client, share.ID, "available", 600)
	if err != nil {
		t.Logf("Failed to get %s share status", share.ID)
		DeleteShare(t, client, share)
		return share, err
	}

	return share, nil
}

// ListShares lists all shares that belong to this tenant's project.
// An error will be returned if the shares could not be listed..
func ListShares(t *testing.T, client *gophercloud.ServiceClient) ([]shares.Share, error) {
	r, err := shares.ListDetail(client, &shares.ListOpts{}).AllPages()
	if err != nil {
		return nil, err
	}

	return shares.ExtractShares(r)
}

// GrantAccess will grant access to an existing share. A fatal error will occur if
// this operation fails.
func GrantAccess(t *testing.T, client *gophercloud.ServiceClient, share *shares.Share) (*shares.AccessRight, error) {
	return shares.GrantAccess(client, share.ID, shares.GrantAccessOpts{
		AccessType:  "ip",
		AccessTo:    "0.0.0.0/32",
		AccessLevel: "ro",
	}).Extract()
}

// RevokeAccess will revoke an exisiting access of a share. A fatal error will occur
// if this operation fails.
func RevokeAccess(t *testing.T, client *gophercloud.ServiceClient, share *shares.Share, accessRight *shares.AccessRight) error {
	return shares.RevokeAccess(client, share.ID, shares.RevokeAccessOpts{
		AccessID: accessRight.ID,
	}).ExtractErr()
}

// GetAccessRightsSlice will retrieve all access rules assigned to a share.
// A fatal error will occur if this operation fails.
func GetAccessRightsSlice(t *testing.T, client *gophercloud.ServiceClient, share *shares.Share) ([]shares.AccessRight, error) {
	return shares.ListAccessRights(client, share.ID).Extract()
}

// DeleteShare will delete a share. A fatal error will occur if the share
// failed to be deleted. This works best when used as a deferred function.
func DeleteShare(t *testing.T, client *gophercloud.ServiceClient, share *shares.Share) {
	err := shares.Delete(client, share.ID).ExtractErr()
	if err != nil {
		t.Errorf("Unable to delete share %s: %v", share.ID, err)
	}

	err = waitForStatus(t, client, share.ID, "deleted", 600)
	if err != nil {
		t.Errorf("Failed to wait for 'deleted' status for %s share: %v", share.ID, err)
	} else {
		t.Logf("Deleted share: %s", share.ID)
	}
}

// ExtendShare extends the capacity of an existing share
func ExtendShare(t *testing.T, client *gophercloud.ServiceClient, share *shares.Share, newSize int) error {
	return shares.Extend(client, share.ID, &shares.ExtendOpts{NewSize: newSize}).ExtractErr()
}

// ShrinkShare shrinks the capacity of an existing share
func ShrinkShare(t *testing.T, client *gophercloud.ServiceClient, share *shares.Share, newSize int) error {
	return shares.Shrink(client, share.ID, &shares.ShrinkOpts{NewSize: newSize}).ExtractErr()
}

func PrintMessages(t *testing.T, c *gophercloud.ServiceClient, id string) error {
	c.Microversion = "2.37"

	allPages, err := messages.List(c, messages.ListOpts{ResourceID: id}).AllPages()
	if err != nil {
		return fmt.Errorf("Unable to retrieve messages: %v", err)
	}

	allMessages, err := messages.ExtractMessages(allPages)
	if err != nil {
		return fmt.Errorf("Unable to extract messages: %v", err)
	}

	for _, message := range allMessages {
		tools.PrintResource(t, message)
	}

	return nil
}

func waitForStatus(t *testing.T, c *gophercloud.ServiceClient, id, status string, secs int) error {
	err := gophercloud.WaitFor(secs, func() (bool, error) {
		current, err := shares.Get(c, id).Extract()
		if err != nil {
			if _, ok := err.(gophercloud.ErrDefault404); ok {
				switch status {
				case "deleted":
					return true, nil
				default:
					return false, err
				}
			}
			return false, err
		}

		if current.Status == status {
			return true, nil
		}

		if strings.Contains(current.Status, "error") {
			return true, fmt.Errorf("An error occurred, wrong status: %s", current.Status)
		}

		return false, nil
	})

	if err != nil {
		mErr := PrintMessages(t, c, id)
		if mErr != nil {
			return fmt.Errorf("Share status is '%s' and unable to get manila messages: %s", err, mErr)
		}
	}

	return err
}
