// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/services"
)

func TestServicesList(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v")
	}

	listOpts := services.ListOpts{
		ServiceType: "identity",
	}

	allPages, err := services.List(client, listOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to list services: %v", err)
	}

	allServices, err := services.ExtractServices(allPages)
	if err != nil {
		t.Fatalf("Unable to extract services: %v", err)
	}

	for _, service := range allServices {
		tools.PrintResource(t, service)
	}

}

func TestServicesCRUD(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v", err)
	}

	createOpts := services.CreateOpts{
		Type: "testing",
		Extra: map[string]interface{}{
			"email": "testservice@example.com",
		},
	}

	// Create service in the default domain
	service, err := CreateService(t, client, &createOpts)
	if err != nil {
		t.Fatalf("Unable to create service: %v", err)
	}
	defer DeleteService(t, client, service.ID)

	tools.PrintResource(t, service)
	tools.PrintResource(t, service.Extra)

	updateOpts := services.UpdateOpts{
		Type: "testing2",
		Extra: map[string]interface{}{
			"description": "Test Users",
			"email":       "thetestservice@example.com",
		},
	}

	newService, err := services.Update(client, service.ID, updateOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to update service: %v", err)
	}

	tools.PrintResource(t, newService)
	tools.PrintResource(t, newService.Extra)
}
