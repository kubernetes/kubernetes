package v2

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/sharedfilesystems/v2/securityservices"
)

// CreateSecurityService will create a security service with a random name. An
// error will be returned if the security service was unable to be created.
func CreateSecurityService(t *testing.T, client *gophercloud.ServiceClient) (*securityservices.SecurityService, error) {
	if testing.Short() {
		t.Skip("Skipping test that requires share network creation in short mode.")
	}

	securityServiceName := tools.RandomString("ACPTTEST", 16)
	t.Logf("Attempting to create security service: %s", securityServiceName)

	createOpts := securityservices.CreateOpts{
		Name: securityServiceName,
		Type: "kerberos",
	}

	securityService, err := securityservices.Create(client, createOpts).Extract()
	if err != nil {
		return securityService, err
	}

	return securityService, nil
}

// DeleteSecurityService will delete a security service. An error will occur if
// the security service was unable to be deleted.
func DeleteSecurityService(t *testing.T, client *gophercloud.ServiceClient, securityService *securityservices.SecurityService) {
	err := securityservices.Delete(client, securityService.ID).ExtractErr()
	if err != nil {
		t.Fatalf("Failed to delete security service %s: %v", securityService.ID, err)
	}

	t.Logf("Deleted security service: %s", securityService.ID)
}

// PrintSecurityService will print a security service and all of its attributes.
func PrintSecurityService(t *testing.T, securityService *securityservices.SecurityService) {
	t.Logf("ID: %s", securityService.ID)
	t.Logf("Project ID: %s", securityService.ProjectID)
	t.Logf("Domain: %s", securityService.Domain)
	t.Logf("Status: %s", securityService.Status)
	t.Logf("Type: %s", securityService.Type)
	t.Logf("Name: %s", securityService.Name)
	t.Logf("Description: %s", securityService.Description)
	t.Logf("DNS IP: %s", securityService.DNSIP)
	t.Logf("User: %s", securityService.User)
	t.Logf("Password: %s", securityService.Password)
	t.Logf("Server: %s", securityService.Server)
	t.Logf("Created at: %v", securityService.CreatedAt)
	t.Logf("Updated at: %v", securityService.UpdatedAt)
}
