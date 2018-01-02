package driverapi

import (
	"fmt"
)

// ErrNoNetwork is returned if no network with the specified id exists
type ErrNoNetwork string

func (enn ErrNoNetwork) Error() string {
	return fmt.Sprintf("No network (%s) exists", string(enn))
}

// NotFound denotes the type of this error
func (enn ErrNoNetwork) NotFound() {}

// ErrEndpointExists is returned if more than one endpoint is added to the network
type ErrEndpointExists string

func (ee ErrEndpointExists) Error() string {
	return fmt.Sprintf("Endpoint (%s) already exists (Only one endpoint allowed)", string(ee))
}

// Forbidden denotes the type of this error
func (ee ErrEndpointExists) Forbidden() {}

// ErrNotImplemented is returned when a Driver has not implemented an API yet
type ErrNotImplemented struct{}

func (eni *ErrNotImplemented) Error() string {
	return "The API is not implemented yet"
}

// NotImplemented denotes the type of this error
func (eni *ErrNotImplemented) NotImplemented() {}

// ErrNoEndpoint is returned if no endpoint with the specified id exists
type ErrNoEndpoint string

func (ene ErrNoEndpoint) Error() string {
	return fmt.Sprintf("No endpoint (%s) exists", string(ene))
}

// NotFound denotes the type of this error
func (ene ErrNoEndpoint) NotFound() {}

// ErrActiveRegistration represents an error when a driver is registered to a networkType that is previously registered
type ErrActiveRegistration string

// Error interface for ErrActiveRegistration
func (ar ErrActiveRegistration) Error() string {
	return fmt.Sprintf("Driver already registered for type %q", string(ar))
}

// Forbidden denotes the type of this error
func (ar ErrActiveRegistration) Forbidden() {}
