// +build acceptance

package v3

import (
	"testing"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/acceptance/tools"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/endpoints"
	"github.com/gophercloud/gophercloud/openstack/identity/v3/services"
)

func TestEndpointsList(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v")
	}

	allPages, err := endpoints.List(client, nil).AllPages()
	if err != nil {
		t.Fatalf("Unable to list endpoints: %v", err)
	}

	allEndpoints, err := endpoints.ExtractEndpoints(allPages)
	if err != nil {
		t.Fatalf("Unable to extract endpoints: %v", err)
	}

	for _, endpoint := range allEndpoints {
		tools.PrintResource(t, endpoint)
	}
}

func TestEndpointsNavigateCatalog(t *testing.T) {
	client, err := clients.NewIdentityV3Client()
	if err != nil {
		t.Fatalf("Unable to obtain an identity client: %v")
	}

	// Discover the service we're interested in.
	serviceListOpts := services.ListOpts{
		ServiceType: "compute",
	}

	allPages, err := services.List(client, serviceListOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to lookup compute service: %v", err)
	}

	allServices, err := services.ExtractServices(allPages)
	if err != nil {
		t.Fatalf("Unable to extract service: %v")
	}

	if len(allServices) != 1 {
		t.Fatalf("Expected one service, got %d", len(allServices))
	}

	computeService := allServices[0]
	tools.PrintResource(t, computeService)

	// Enumerate the endpoints available for this service.
	endpointListOpts := endpoints.ListOpts{
		Availability: gophercloud.AvailabilityPublic,
		ServiceID:    computeService.ID,
	}

	allPages, err = endpoints.List(client, endpointListOpts).AllPages()
	if err != nil {
		t.Fatalf("Unable to lookup compute endpoint: %v", err)
	}

	allEndpoints, err := endpoints.ExtractEndpoints(allPages)
	if err != nil {
		t.Fatalf("Unable to extract endpoint: %v")
	}

	if len(allEndpoints) != 1 {
		t.Fatalf("Expected one endpoint, got %d", len(allEndpoints))
	}

	tools.PrintResource(t, allEndpoints[0])

}
