// +build acceptance

package v3

import (
	"testing"

	services3 "github.com/rackspace/gophercloud/openstack/identity/v3/services"
	"github.com/rackspace/gophercloud/pagination"
)

func TestListServices(t *testing.T) {
	// Create a service client.
	serviceClient := createAuthenticatedClient(t)
	if serviceClient == nil {
		return
	}

	// Use the client to list all available services.
	pager := services3.List(serviceClient, services3.ListOpts{})
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		parts, err := services3.ExtractServices(page)
		if err != nil {
			return false, err
		}

		t.Logf("--- Page ---")
		for _, service := range parts {
			t.Logf("Service: %32s %15s %10s %s", service.ID, service.Type, service.Name, *service.Description)
		}
		return true, nil
	})
	if err != nil {
		t.Errorf("Unexpected error traversing pages: %v", err)
	}
}
