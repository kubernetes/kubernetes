package openstack

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	tokens2 "github.com/rackspace/gophercloud/openstack/identity/v2/tokens"
	endpoints3 "github.com/rackspace/gophercloud/openstack/identity/v3/endpoints"
	services3 "github.com/rackspace/gophercloud/openstack/identity/v3/services"
	"github.com/rackspace/gophercloud/pagination"
)

// V2EndpointURL discovers the endpoint URL for a specific service from a ServiceCatalog acquired
// during the v2 identity service. The specified EndpointOpts are used to identify a unique,
// unambiguous endpoint to return. It's an error both when multiple endpoints match the provided
// criteria and when none do. The minimum that can be specified is a Type, but you will also often
// need to specify a Name and/or a Region depending on what's available on your OpenStack
// deployment.
func V2EndpointURL(catalog *tokens2.ServiceCatalog, opts gophercloud.EndpointOpts) (string, error) {
	// Extract Endpoints from the catalog entries that match the requested Type, Name if provided, and Region if provided.
	var endpoints = make([]tokens2.Endpoint, 0, 1)
	for _, entry := range catalog.Entries {
		if (entry.Type == opts.Type) && (opts.Name == "" || entry.Name == opts.Name) {
			for _, endpoint := range entry.Endpoints {
				if opts.Region == "" || endpoint.Region == opts.Region {
					endpoints = append(endpoints, endpoint)
				}
			}
		}
	}

	// Report an error if the options were ambiguous.
	if len(endpoints) > 1 {
		return "", fmt.Errorf("Discovered %d matching endpoints: %#v", len(endpoints), endpoints)
	}

	// Extract the appropriate URL from the matching Endpoint.
	for _, endpoint := range endpoints {
		switch opts.Availability {
		case gophercloud.AvailabilityPublic:
			return gophercloud.NormalizeURL(endpoint.PublicURL), nil
		case gophercloud.AvailabilityInternal:
			return gophercloud.NormalizeURL(endpoint.InternalURL), nil
		case gophercloud.AvailabilityAdmin:
			return gophercloud.NormalizeURL(endpoint.AdminURL), nil
		default:
			return "", fmt.Errorf("Unexpected availability in endpoint query: %s", opts.Availability)
		}
	}

	// Report an error if there were no matching endpoints.
	return "", gophercloud.ErrEndpointNotFound
}

// V3EndpointURL discovers the endpoint URL for a specific service using multiple calls against
// an identity v3 service endpoint. The specified EndpointOpts are used to identify a unique,
// unambiguous endpoint to return. It's an error both when multiple endpoints match the provided
// criteria and when none do. The minimum that can be specified is a Type, but you will also often
// need to specify a Name and/or a Region depending on what's available on your OpenStack
// deployment.
func V3EndpointURL(v3Client *gophercloud.ServiceClient, opts gophercloud.EndpointOpts) (string, error) {
	// Discover the service we're interested in.
	var services = make([]services3.Service, 0, 1)
	servicePager := services3.List(v3Client, services3.ListOpts{ServiceType: opts.Type})
	err := servicePager.EachPage(func(page pagination.Page) (bool, error) {
		part, err := services3.ExtractServices(page)
		if err != nil {
			return false, err
		}

		for _, service := range part {
			if service.Name == opts.Name {
				services = append(services, service)
			}
		}

		return true, nil
	})
	if err != nil {
		return "", err
	}

	if len(services) == 0 {
		return "", gophercloud.ErrServiceNotFound
	}
	if len(services) > 1 {
		return "", fmt.Errorf("Discovered %d matching services: %#v", len(services), services)
	}
	service := services[0]

	// Enumerate the endpoints available for this service.
	var endpoints []endpoints3.Endpoint
	endpointPager := endpoints3.List(v3Client, endpoints3.ListOpts{
		Availability: opts.Availability,
		ServiceID:    service.ID,
	})
	err = endpointPager.EachPage(func(page pagination.Page) (bool, error) {
		part, err := endpoints3.ExtractEndpoints(page)
		if err != nil {
			return false, err
		}

		for _, endpoint := range part {
			if opts.Region == "" || endpoint.Region == opts.Region {
				endpoints = append(endpoints, endpoint)
			}
		}

		return true, nil
	})
	if err != nil {
		return "", err
	}

	if len(endpoints) == 0 {
		return "", gophercloud.ErrEndpointNotFound
	}
	if len(endpoints) > 1 {
		return "", fmt.Errorf("Discovered %d matching endpoints: %#v", len(endpoints), endpoints)
	}
	endpoint := endpoints[0]

	return gophercloud.NormalizeURL(endpoint.URL), nil
}
