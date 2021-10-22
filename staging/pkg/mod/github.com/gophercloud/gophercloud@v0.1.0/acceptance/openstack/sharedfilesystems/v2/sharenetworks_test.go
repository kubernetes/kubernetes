package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/sharenetworks"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestShareNetworkCreateDestroy(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create shared file system client: %v", err)
	}

	shareNetwork, err := CreateShareNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create share network: %v", err)
	}

	newShareNetwork, err := sharenetworks.Get(client, shareNetwork.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve shareNetwork: %v", err)
	}

	if newShareNetwork.Name != shareNetwork.Name {
		t.Fatalf("Share network name was expeted to be: %s", shareNetwork.Name)
	}

	if newShareNetwork.Description != shareNetwork.Description {
		t.Fatalf("Share network description was expeted to be: %s", shareNetwork.Description)
	}

	tools.PrintResource(t, shareNetwork)

	defer DeleteShareNetwork(t, client, shareNetwork.ID)
}

// Create a share network and update the name and description. Get the share
// network and verify that the name and description have been updated
func TestShareNetworkUpdate(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create shared file system client: %v", err)
	}

	shareNetwork, err := CreateShareNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create share network: %v", err)
	}

	expectedShareNetwork, err := sharenetworks.Get(client, shareNetwork.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve shareNetwork: %v", err)
	}

	name := "NewName"
	description := ""
	options := sharenetworks.UpdateOpts{
		Name:        &name,
		Description: &description,
	}

	expectedShareNetwork.Name = name
	expectedShareNetwork.Description = description

	_, err = sharenetworks.Update(client, shareNetwork.ID, options).Extract()
	if err != nil {
		t.Errorf("Unable to update shareNetwork: %v", err)
	}

	updatedShareNetwork, err := sharenetworks.Get(client, shareNetwork.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve shareNetwork: %v", err)
	}

	// Update time has to be set in order to get the assert equal to pass
	expectedShareNetwork.UpdatedAt = updatedShareNetwork.UpdatedAt

	th.CheckDeepEquals(t, expectedShareNetwork, updatedShareNetwork)

	tools.PrintResource(t, shareNetwork)

	defer DeleteShareNetwork(t, client, shareNetwork.ID)
}

func TestShareNetworkListDetail(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	allPages, err := sharenetworks.ListDetail(client, sharenetworks.ListOpts{}).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve share networks: %v", err)
	}

	allShareNetworks, err := sharenetworks.ExtractShareNetworks(allPages)
	if err != nil {
		t.Fatalf("Unable to extract share networks: %v", err)
	}

	for _, shareNetwork := range allShareNetworks {
		tools.PrintResource(t, &shareNetwork)
	}
}

// The test creates 2 shared networks and verifies that only the one(s) with
// a particular name are being listed
func TestShareNetworkListFiltering(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	shareNetwork, err := CreateShareNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create share network: %v", err)
	}
	defer DeleteShareNetwork(t, client, shareNetwork.ID)

	shareNetwork, err = CreateShareNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create share network: %v", err)
	}
	defer DeleteShareNetwork(t, client, shareNetwork.ID)

	options := sharenetworks.ListOpts{
		Name: shareNetwork.Name,
	}

	allPages, err := sharenetworks.ListDetail(client, options).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve share networks: %v", err)
	}

	allShareNetworks, err := sharenetworks.ExtractShareNetworks(allPages)
	if err != nil {
		t.Fatalf("Unable to extract share networks: %v", err)
	}

	for _, listedShareNetwork := range allShareNetworks {
		if listedShareNetwork.Name != shareNetwork.Name {
			t.Fatalf("The name of the share network was expected to be %s", shareNetwork.Name)
		}
		tools.PrintResource(t, &listedShareNetwork)
	}
}

func TestShareNetworkListPagination(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	shareNetwork, err := CreateShareNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create share network: %v", err)
	}
	defer DeleteShareNetwork(t, client, shareNetwork.ID)

	shareNetwork, err = CreateShareNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create share network: %v", err)
	}
	defer DeleteShareNetwork(t, client, shareNetwork.ID)

	count := 0

	err = sharenetworks.ListDetail(client, sharenetworks.ListOpts{Offset: 0, Limit: 1}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		_, err := sharenetworks.ExtractShareNetworks(page)
		if err != nil {
			t.Fatalf("Failed to extract share networks: %v", err)
			return false, err
		}

		return true, nil
	})
	if err != nil {
		t.Fatalf("Unable to retrieve share networks: %v", err)
	}

	if count < 2 {
		t.Fatal("Expected to get at least 2 pages")
	}

}

func TestShareNetworkAddRemoveSecurityService(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	securityService, err := CreateSecurityService(t, client)
	if err != nil {
		t.Fatalf("Unable to create security service: %v", err)
	}
	defer DeleteSecurityService(t, client, securityService)

	shareNetwork, err := CreateShareNetwork(t, client)
	if err != nil {
		t.Fatalf("Unable to create share network: %v", err)
	}
	defer DeleteShareNetwork(t, client, shareNetwork.ID)

	options := sharenetworks.AddSecurityServiceOpts{
		SecurityServiceID: securityService.ID,
	}

	_, err = sharenetworks.AddSecurityService(client, shareNetwork.ID, options).Extract()
	if err != nil {
		t.Errorf("Unable to add security service: %v", err)
	}

	removeOptions := sharenetworks.RemoveSecurityServiceOpts{
		SecurityServiceID: securityService.ID,
	}

	_, err = sharenetworks.RemoveSecurityService(client, shareNetwork.ID, removeOptions).Extract()
	if err != nil {
		t.Errorf("Unable to remove security service: %v", err)
	}

	tools.PrintResource(t, shareNetwork)
}
