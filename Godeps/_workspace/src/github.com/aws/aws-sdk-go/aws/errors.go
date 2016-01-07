package aws

import "github.com/aws/aws-sdk-go/aws/awserr"

var (
	// ErrMissingRegion is an error that is returned if region configuration is
	// not found.
	//
	// @readonly
	ErrMissingRegion = awserr.New("MissingRegion", "could not find region configuration", nil)

	// ErrMissingEndpoint is an error that is returned if an endpoint cannot be
	// resolved for a service.
	//
	// @readonly
	ErrMissingEndpoint = awserr.New("MissingEndpoint", "'Endpoint' configuration is required for this service", nil)
)
