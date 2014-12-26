// +build acceptance

package v3

import (
	"testing"

	"github.com/rackspace/gophercloud"
	endpoints3 "github.com/rackspace/gophercloud/openstack/identity/v3/endpoints"
	services3 "github.com/rackspace/gophercloud/openstack/identity/v3/services"
	"github.com/rackspace/gophercloud/pagination"
)

func TestListEndpoints(t *testing.T) {
	// Create a service client.
	serviceClient := createAuthenticatedClient(t)
	if serviceClient == nil {
		return
	}

	// Use the service to list all available endpoints.
	pager := endpoints3.List(serviceClient, endpoints3.ListOpts{})
	err := pager.EachPage(func(page pagination.Page) (bool, error) {
		t.Logf("--- Page ---")

		endpoints, err := endpoints3.ExtractEndpoints(page)
		if err != nil {
			t.Fatalf("Error extracting endpoings: %v", err)
		}

		for _, endpoint := range endpoints {
			t.Logf("Endpoint: %8s %10s %9s %s",
				endpoint.ID,
				endpoint.Availability,
				endpoint.Name,
				endpoint.URL)
		}

		return true, nil
	})
	if err != nil {
		t.Errorf("Unexpected error while iterating endpoint pages: %v", err)
	}
}

func TestNavigateCatalog(t *testing.T) {
	// Create a service client.
	client := createAuthenticatedClient(t)
	if client == nil {
		return
	}

	var compute *services3.Service
	var endpoint *endpoints3.Endpoint

	// Discover the service we're interested in.
	servicePager := services3.List(client, services3.ListOpts{ServiceType: "compute"})
	err := servicePager.EachPage(func(page pagination.Page) (bool, error) {
		part, err := services3.ExtractServices(page)
		if err != nil {
			return false, err
		}
		if compute != nil {
			t.Fatalf("Expected one service, got more than one page")
			return false, nil
		}
		if len(part) != 1 {
			t.Fatalf("Expected one service, got %d", len(part))
			return false, nil
		}

		compute = &part[0]
		return true, nil
	})
	if err != nil {
		t.Fatalf("Unexpected error iterating pages: %v", err)
	}

	if compute == nil {
		t.Fatalf("No compute service found.")
	}

	// Enumerate the endpoints available for this service.
	computePager := endpoints3.List(client, endpoints3.ListOpts{
		Availability: gophercloud.AvailabilityPublic,
		ServiceID:    compute.ID,
	})
	err = computePager.EachPage(func(page pagination.Page) (bool, error) {
		part, err := endpoints3.ExtractEndpoints(page)
		if err != nil {
			return false, err
		}
		if endpoint != nil {
			t.Fatalf("Expected one endpoint, got more than one page")
			return false, nil
		}
		if len(part) != 1 {
			t.Fatalf("Expected one endpoint, got %d", len(part))
			return false, nil
		}

		endpoint = &part[0]
		return true, nil
	})

	if endpoint == nil {
		t.Fatalf("No endpoint found.")
	}

	t.Logf("Success. The compute endpoint is at %s.", endpoint.URL)
}
