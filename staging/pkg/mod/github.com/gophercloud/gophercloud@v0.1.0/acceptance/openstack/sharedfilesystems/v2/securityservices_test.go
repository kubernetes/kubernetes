package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/securityservices"
)

func TestSecurityServiceCreateDelete(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create shared file system client: %v", err)
	}

	securityService, err := CreateSecurityService(t, client)
	if err != nil {
		t.Fatalf("Unable to create security service: %v", err)
	}

	newSecurityService, err := securityservices.Get(client, securityService.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve the security service: %v", err)
	}

	if newSecurityService.Name != securityService.Name {
		t.Fatalf("Security service name was expeted to be: %s", securityService.Name)
	}

	if newSecurityService.Description != securityService.Description {
		t.Fatalf("Security service description was expeted to be: %s", securityService.Description)
	}

	tools.PrintResource(t, securityService)

	defer DeleteSecurityService(t, client, securityService)
}

func TestSecurityServiceList(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	allPages, err := securityservices.List(client, securityservices.ListOpts{}).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve security services: %v", err)
	}

	allSecurityServices, err := securityservices.ExtractSecurityServices(allPages)
	if err != nil {
		t.Fatalf("Unable to extract security services: %v", err)
	}

	for _, securityService := range allSecurityServices {
		tools.PrintResource(t, &securityService)
	}
}

// The test creates 2 security services and verifies that only the one(s) with
// a particular name are being listed
func TestSecurityServiceListFiltering(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create a shared file system client: %v", err)
	}

	securityService, err := CreateSecurityService(t, client)
	if err != nil {
		t.Fatalf("Unable to create security service: %v", err)
	}
	defer DeleteSecurityService(t, client, securityService)

	securityService, err = CreateSecurityService(t, client)
	if err != nil {
		t.Fatalf("Unable to create security service: %v", err)
	}
	defer DeleteSecurityService(t, client, securityService)

	options := securityservices.ListOpts{
		Name: securityService.Name,
	}

	allPages, err := securityservices.List(client, options).AllPages()
	if err != nil {
		t.Fatalf("Unable to retrieve security services: %v", err)
	}

	allSecurityServices, err := securityservices.ExtractSecurityServices(allPages)
	if err != nil {
		t.Fatalf("Unable to extract security services: %v", err)
	}

	for _, listedSecurityService := range allSecurityServices {
		if listedSecurityService.Name != securityService.Name {
			t.Fatalf("The name of the security service was expected to be %s", securityService.Name)
		}
		tools.PrintResource(t, &listedSecurityService)
	}
}

// Create a security service and update the name and description. Get the security
// service and verify that the name and description have been updated
func TestSecurityServiceUpdate(t *testing.T) {
	client, err := clients.NewSharedFileSystemV2Client()
	if err != nil {
		t.Fatalf("Unable to create shared file system client: %v", err)
	}

	securityService, err := CreateSecurityService(t, client)
	if err != nil {
		t.Fatalf("Unable to create security service: %v", err)
	}

	name := "NewName"
	description := ""
	options := securityservices.UpdateOpts{
		Name:        &name,
		Description: &description,
		Type:        "ldap",
	}

	_, err = securityservices.Update(client, securityService.ID, options).Extract()
	if err != nil {
		t.Errorf("Unable to update the security service: %v", err)
	}

	newSecurityService, err := securityservices.Get(client, securityService.ID).Extract()
	if err != nil {
		t.Errorf("Unable to retrieve the security service: %v", err)
	}

	if newSecurityService.Name != name {
		t.Fatalf("Security service name was expeted to be: %s", name)
	}

	if newSecurityService.Description != description {
		t.Fatalf("Security service description was expeted to be: %s", description)
	}

	if newSecurityService.Type != options.Type {
		t.Fatalf("Security service type was expected to be: %s", options.Type)
	}

	tools.PrintResource(t, securityService)

	defer DeleteSecurityService(t, client, securityService)
}
