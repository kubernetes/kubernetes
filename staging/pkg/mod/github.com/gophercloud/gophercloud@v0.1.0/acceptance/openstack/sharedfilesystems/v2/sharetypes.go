package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/sharetypes"
)

// CreateShareType will create a share type with a random name. An
// error will be returned if the share type was unable to be created.
func CreateShareType(t *testing.T, client *gophercloud.ServiceClient) (*sharetypes.ShareType, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires share type creation in short mode.")
	}

	shareTypeName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create share type: %s", shareTypeName)

	extraSpecsOps := sharetypes.ExtraSpecsOpts{
		DriverHandlesShareServers: true,
	}

	createOpts := sharetypes.CreateOpts{
		Name:       shareTypeName,
		IsPublic:   false,
		ExtraSpecs: extraSpecsOps,
	}

	shareType, err := sharetypes.Create(client, createOpts).Extract()
	if err != nil {
		return shareType, err
	}

	return shareType, nil
}

// DeleteShareType will delete a share type. An error will occur if
// the share type was unable to be deleted.
func DeleteShareType(t *testing.T, client *gophercloud.ServiceClient, shareType *sharetypes.ShareType) {
	err := sharetypes.Delete(client, shareType.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Failed to delete share type %s: %v", shareType.ID, err)
	}

	t.Logf("Deleted share type: %s", shareType.ID)
}
