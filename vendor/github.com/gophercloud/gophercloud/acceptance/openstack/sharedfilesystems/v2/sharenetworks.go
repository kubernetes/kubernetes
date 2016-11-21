package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/sharenetworks"
)

// CreateShareNetwork will create a share network with a random name. An
// error will be returned if the share network was unable to be created.
func CreateShareNetwork(t *testing.T, client *gophercloud.ServiceClient) (*sharenetworks.ShareNetwork, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires share network creation in short mode.")
	}

	shareNetworkName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create share network: %s", shareNetworkName)

	createOpts := sharenetworks.CreateOpts{
		Name:        shareNetworkName,
		Description: "This is a shared network",
	}

	shareNetwork, err := sharenetworks.Create(client, createOpts).Extract()
	if err != nil {
		return shareNetwork, err
	}

	return shareNetwork, nil
}

// DeleteShareNetwork will delete a share network. An error will occur if
// the share network was unable to be deleted.
func DeleteShareNetwork(t *testing.T, client *gophercloud.ServiceClient, shareNetwork *sharenetworks.ShareNetwork) {
	err := sharenetworks.Delete(client, shareNetwork.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Failed to delete share network %s: %v", shareNetwork.ID, err)
	}

	t.Logf("Deleted share network: %s", shareNetwork.ID)
}

// PrintShareNetwork will print a share network and all of its attributes.
func PrintShareNetwork(t *testing.T, sharenetwork *sharenetworks.ShareNetwork) {
	t.Logf("ID: %s", sharenetwork.ID)
	t.Logf("Project ID: %s", sharenetwork.ProjectID)
	t.Logf("Neutron network ID: %s", sharenetwork.NeutronNetID)
	t.Logf("Neutron sub-network ID: %s", sharenetwork.NeutronSubnetID)
	t.Logf("Nova network ID: %s", sharenetwork.NovaNetID)
	t.Logf("Network type: %s", sharenetwork.NetworkType)
	t.Logf("Segmentation ID: %d", sharenetwork.SegmentationID)
	t.Logf("CIDR: %s", sharenetwork.CIDR)
	t.Logf("IP version: %d", sharenetwork.IPVersion)
	t.Logf("Name: %s", sharenetwork.Name)
	t.Logf("Description: %s", sharenetwork.Description)
	t.Logf("Created at: %v", sharenetwork.CreatedAt)
	t.Logf("Updated at: %v", sharenetwork.UpdatedAt)
}
