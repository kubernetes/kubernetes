package gophercloud

import "errors"

var (
	// ErrServiceNotFound is returned when no service in a service catalog matches
	// the provided EndpointOpts. This is generally returned by provider service
	// factory methods like "NewComputeV2()" and can mean that a service is not
	// enabled for your account.
	ErrServiceNotFound = errors.New("No suitable service could be found in the service catalog.")

	// ErrEndpointNotFound is returned when no available endpoints match the
	// provided EndpointOpts. This is also generally returned by provider service
	// factory methods, and usually indicates that a region was specified
	// incorrectly.
	ErrEndpointNotFound = errors.New("No suitable endpoint could be found in the service catalog.")
)

// Availability indicates to whom a specific service endpoint is accessible:
// the internet at large, internal networks only, or only to administrators.
// Different identity services use different terminology for these. Identity v2
// lists them as different kinds of URLs within the service catalog ("adminURL",
// "internalURL", and "publicURL"), while v3 lists them as "Interfaces" in an
// endpoint's response.
type Availability string

const (
	// AvailabilityAdmin indicates that an endpoint is only available to
	// administrators.
	AvailabilityAdmin Availability = "admin"

	// AvailabilityPublic indicates that an endpoint is available to everyone on
	// the internet.
	AvailabilityPublic Availability = "public"

	// AvailabilityInternal indicates that an endpoint is only available within
	// the cluster's internal network.
	AvailabilityInternal Availability = "internal"
)

// EndpointOpts specifies search criteria used by queries against an
// OpenStack service catalog. The options must contain enough information to
// unambiguously identify one, and only one, endpoint within the catalog.
//
// Usually, these are passed to service client factory functions in a provider
// package, like "rackspace.NewComputeV2()".
type EndpointOpts struct {
	// Type [required] is the service type for the client (e.g., "compute",
	// "object-store"). Generally, this will be supplied by the service client
	// function, but a user-given value will be honored if provided.
	Type string

	// Name [optional] is the service name for the client (e.g., "nova") as it
	// appears in the service catalog. Services can have the same Type but a
	// different Name, which is why both Type and Name are sometimes needed.
	Name string

	// Region [required] is the geographic region in which the endpoint resides,
	// generally specifying which datacenter should house your resources.
	// Required only for services that span multiple regions.
	Region string

	// Availability [optional] is the visibility of the endpoint to be returned.
	// Valid types include the constants AvailabilityPublic, AvailabilityInternal,
	// or AvailabilityAdmin from this package.
	//
	// Availability is not required, and defaults to AvailabilityPublic. Not all
	// providers or services offer all Availability options.
	Availability Availability
}

/*
EndpointLocator is an internal function to be used by provider implementations.

It provides an implementation that locates a single endpoint from a service
catalog for a specific ProviderClient based on user-provided EndpointOpts. The
provider then uses it to discover related ServiceClients.
*/
type EndpointLocator func(EndpointOpts) (string, error)

// ApplyDefaults is an internal method to be used by provider implementations.
//
// It sets EndpointOpts fields if not already set, including a default type.
// Currently, EndpointOpts.Availability defaults to the public endpoint.
func (eo *EndpointOpts) ApplyDefaults(t string) {
	if eo.Type == "" {
		eo.Type = t
	}
	if eo.Availability == "" {
		eo.Availability = AvailabilityPublic
	}
}
