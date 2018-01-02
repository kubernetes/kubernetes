package bridge

import "fmt"

// ErrInvalidEndpointConfig error is returned when an endpoint create is attempted with an invalid endpoint configuration.
type ErrInvalidEndpointConfig struct{}

func (eiec *ErrInvalidEndpointConfig) Error() string {
	return "trying to create an endpoint with an invalid endpoint configuration"
}

// BadRequest denotes the type of this error
func (eiec *ErrInvalidEndpointConfig) BadRequest() {}

// ErrNoIPAddr error is returned when bridge has no IPv4 address configured.
type ErrNoIPAddr struct{}

func (enip *ErrNoIPAddr) Error() string {
	return "bridge has no IPv4 address configured"
}

// InternalError denotes the type of this error
func (enip *ErrNoIPAddr) InternalError() {}

// ErrInvalidGateway is returned when the user provided default gateway (v4/v6) is not not valid.
type ErrInvalidGateway struct{}

func (eig *ErrInvalidGateway) Error() string {
	return "default gateway ip must be part of the network"
}

// BadRequest denotes the type of this error
func (eig *ErrInvalidGateway) BadRequest() {}

// ErrInvalidMtu is returned when the user provided MTU is not valid.
type ErrInvalidMtu int

func (eim ErrInvalidMtu) Error() string {
	return fmt.Sprintf("invalid MTU number: %d", int(eim))
}

// BadRequest denotes the type of this error
func (eim ErrInvalidMtu) BadRequest() {}

// ErrUnsupportedAddressType is returned when the specified address type is not supported.
type ErrUnsupportedAddressType string

func (uat ErrUnsupportedAddressType) Error() string {
	return fmt.Sprintf("unsupported address type: %s", string(uat))
}

// BadRequest denotes the type of this error
func (uat ErrUnsupportedAddressType) BadRequest() {}

// ActiveEndpointsError is returned when there are
// still active endpoints in the network being deleted.
type ActiveEndpointsError string

func (aee ActiveEndpointsError) Error() string {
	return fmt.Sprintf("network %s has active endpoint", string(aee))
}

// Forbidden denotes the type of this error
func (aee ActiveEndpointsError) Forbidden() {}

// InvalidNetworkIDError is returned when the passed
// network id for an existing network is not a known id.
type InvalidNetworkIDError string

func (inie InvalidNetworkIDError) Error() string {
	return fmt.Sprintf("invalid network id %s", string(inie))
}

// NotFound denotes the type of this error
func (inie InvalidNetworkIDError) NotFound() {}

// InvalidEndpointIDError is returned when the passed
// endpoint id is not valid.
type InvalidEndpointIDError string

func (ieie InvalidEndpointIDError) Error() string {
	return fmt.Sprintf("invalid endpoint id: %s", string(ieie))
}

// BadRequest denotes the type of this error
func (ieie InvalidEndpointIDError) BadRequest() {}

// EndpointNotFoundError is returned when the no endpoint
// with the passed endpoint id is found.
type EndpointNotFoundError string

func (enfe EndpointNotFoundError) Error() string {
	return fmt.Sprintf("endpoint not found: %s", string(enfe))
}

// NotFound denotes the type of this error
func (enfe EndpointNotFoundError) NotFound() {}

// NonDefaultBridgeExistError is returned when a non-default
// bridge config is passed but it does not already exist.
type NonDefaultBridgeExistError string

func (ndbee NonDefaultBridgeExistError) Error() string {
	return fmt.Sprintf("bridge device with non default name %s must be created manually", string(ndbee))
}

// Forbidden denotes the type of this error
func (ndbee NonDefaultBridgeExistError) Forbidden() {}

// NonDefaultBridgeNeedsIPError is returned when a non-default
// bridge config is passed but it has no ip configured
type NonDefaultBridgeNeedsIPError string

func (ndbee NonDefaultBridgeNeedsIPError) Error() string {
	return fmt.Sprintf("bridge device with non default name %s must have a valid IP address", string(ndbee))
}

// Forbidden denotes the type of this error
func (ndbee NonDefaultBridgeNeedsIPError) Forbidden() {}
