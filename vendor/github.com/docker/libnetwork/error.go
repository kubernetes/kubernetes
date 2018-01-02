package libnetwork

import (
	"fmt"
)

// ErrNoSuchNetwork is returned when a network query finds no result
type ErrNoSuchNetwork string

func (nsn ErrNoSuchNetwork) Error() string {
	return fmt.Sprintf("network %s not found", string(nsn))
}

// NotFound denotes the type of this error
func (nsn ErrNoSuchNetwork) NotFound() {}

// ErrNoSuchEndpoint is returned when an endpoint query finds no result
type ErrNoSuchEndpoint string

func (nse ErrNoSuchEndpoint) Error() string {
	return fmt.Sprintf("endpoint %s not found", string(nse))
}

// NotFound denotes the type of this error
func (nse ErrNoSuchEndpoint) NotFound() {}

// ErrInvalidNetworkDriver is returned if an invalid driver
// name is passed.
type ErrInvalidNetworkDriver string

func (ind ErrInvalidNetworkDriver) Error() string {
	return fmt.Sprintf("invalid driver bound to network: %s", string(ind))
}

// BadRequest denotes the type of this error
func (ind ErrInvalidNetworkDriver) BadRequest() {}

// ErrInvalidJoin is returned if a join is attempted on an endpoint
// which already has a container joined.
type ErrInvalidJoin struct{}

func (ij ErrInvalidJoin) Error() string {
	return "a container has already joined the endpoint"
}

// BadRequest denotes the type of this error
func (ij ErrInvalidJoin) BadRequest() {}

// ErrNoContainer is returned when the endpoint has no container
// attached to it.
type ErrNoContainer struct{}

func (nc ErrNoContainer) Error() string {
	return "no container is attached to the endpoint"
}

// Maskable denotes the type of this error
func (nc ErrNoContainer) Maskable() {}

// ErrInvalidID is returned when a query-by-id method is being invoked
// with an empty id parameter
type ErrInvalidID string

func (ii ErrInvalidID) Error() string {
	return fmt.Sprintf("invalid id: %s", string(ii))
}

// BadRequest denotes the type of this error
func (ii ErrInvalidID) BadRequest() {}

// ErrInvalidName is returned when a query-by-name or resource create method is
// invoked with an empty name parameter
type ErrInvalidName string

func (in ErrInvalidName) Error() string {
	return fmt.Sprintf("invalid name: %s", string(in))
}

// BadRequest denotes the type of this error
func (in ErrInvalidName) BadRequest() {}

// ErrInvalidConfigFile type is returned when an invalid LibNetwork config file is detected
type ErrInvalidConfigFile string

func (cf ErrInvalidConfigFile) Error() string {
	return fmt.Sprintf("Invalid Config file %q", string(cf))
}

// NetworkTypeError type is returned when the network type string is not
// known to libnetwork.
type NetworkTypeError string

func (nt NetworkTypeError) Error() string {
	return fmt.Sprintf("unknown driver %q", string(nt))
}

// NotFound denotes the type of this error
func (nt NetworkTypeError) NotFound() {}

// NetworkNameError is returned when a network with the same name already exists.
type NetworkNameError string

func (nnr NetworkNameError) Error() string {
	return fmt.Sprintf("network with name %s already exists", string(nnr))
}

// Forbidden denotes the type of this error
func (nnr NetworkNameError) Forbidden() {}

// UnknownNetworkError is returned when libnetwork could not find in its database
// a network with the same name and id.
type UnknownNetworkError struct {
	name string
	id   string
}

func (une *UnknownNetworkError) Error() string {
	return fmt.Sprintf("unknown network %s id %s", une.name, une.id)
}

// NotFound denotes the type of this error
func (une *UnknownNetworkError) NotFound() {}

// ActiveEndpointsError is returned when a network is deleted which has active
// endpoints in it.
type ActiveEndpointsError struct {
	name string
	id   string
}

func (aee *ActiveEndpointsError) Error() string {
	return fmt.Sprintf("network %s id %s has active endpoints", aee.name, aee.id)
}

// Forbidden denotes the type of this error
func (aee *ActiveEndpointsError) Forbidden() {}

// UnknownEndpointError is returned when libnetwork could not find in its database
// an endpoint with the same name and id.
type UnknownEndpointError struct {
	name string
	id   string
}

func (uee *UnknownEndpointError) Error() string {
	return fmt.Sprintf("unknown endpoint %s id %s", uee.name, uee.id)
}

// NotFound denotes the type of this error
func (uee *UnknownEndpointError) NotFound() {}

// ActiveContainerError is returned when an endpoint is deleted which has active
// containers attached to it.
type ActiveContainerError struct {
	name string
	id   string
}

func (ace *ActiveContainerError) Error() string {
	return fmt.Sprintf("endpoint with name %s id %s has active containers", ace.name, ace.id)
}

// Forbidden denotes the type of this error
func (ace *ActiveContainerError) Forbidden() {}

// InvalidContainerIDError is returned when an invalid container id is passed
// in Join/Leave
type InvalidContainerIDError string

func (id InvalidContainerIDError) Error() string {
	return fmt.Sprintf("invalid container id %s", string(id))
}

// BadRequest denotes the type of this error
func (id InvalidContainerIDError) BadRequest() {}

// ManagerRedirectError is returned when the request should be redirected to Manager
type ManagerRedirectError string

func (mr ManagerRedirectError) Error() string {
	return "Redirect the request to the manager"
}

// Maskable denotes the type of this error
func (mr ManagerRedirectError) Maskable() {}

// ErrDataStoreNotInitialized is returned if an invalid data scope is passed
// for getting data store
type ErrDataStoreNotInitialized string

func (dsni ErrDataStoreNotInitialized) Error() string {
	return fmt.Sprintf("datastore for scope %q is not initialized", string(dsni))
}
