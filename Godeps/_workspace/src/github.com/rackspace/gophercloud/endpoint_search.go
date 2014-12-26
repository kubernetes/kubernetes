package gophercloud

import "errors"

var (
	// ErrServiceNotFound is returned when no service matches the EndpointOpts.
	ErrServiceNotFound = errors.New("No suitable service could be found in the service catalog.")

	// ErrEndpointNotFound is returned when no available endpoints match the provided EndpointOpts.
	ErrEndpointNotFound = errors.New("No suitable endpoint could be found in the service catalog.")
)

// Availability indicates whether a specific service endpoint is accessible.
// Identity v2 lists these as different kinds of URLs ("adminURL",
// "internalURL", and "publicURL"), while v3 lists them as "Interfaces".
type Availability string

const (
	// AvailabilityAdmin makes an endpoint only available to administrators.
	AvailabilityAdmin Availability = "admin"

	// AvailabilityPublic makes an endpoint available to everyone.
	AvailabilityPublic Availability = "public"

	// AvailabilityInternal makes an endpoint only available within the cluster.
	AvailabilityInternal Availability = "internal"
)

// EndpointOpts contains options for finding an endpoint for an Openstack client.
type EndpointOpts struct {
	// Type is the service type for the client (e.g., "compute", "object-store").
	// Required.
	Type string

	// Name is the service name for the client (e.g., "nova") as it appears in
	// the service catalog. Services can have the same Type but a different Name,
	// which is why both Type and Name are sometimes needed. Optional.
	Name string

	// Region is the geographic region in which the service resides. Required only
	// for services that span multiple regions.
	Region string

	// Availability is the visibility of the endpoint to be returned. Valid types
	// are: AvailabilityPublic, AvailabilityInternal, or AvailabilityAdmin.
	// Availability is not required, and defaults to AvailabilityPublic.
	// Not all providers or services offer all Availability options.
	Availability Availability
}

// EndpointLocator is a function that describes how to locate a single endpoint
// from a service catalog for a specific ProviderClient. It should be set
// during ProviderClient authentication and used to discover related ServiceClients.
type EndpointLocator func(EndpointOpts) (string, error)

// ApplyDefaults sets EndpointOpts fields if not already set. Currently,
// EndpointOpts.Availability defaults to the public endpoint.
func (eo *EndpointOpts) ApplyDefaults(t string) {
	if eo.Type == "" {
		eo.Type = t
	}
	if eo.Availability == "" {
		eo.Availability = AvailabilityPublic
	}
}
