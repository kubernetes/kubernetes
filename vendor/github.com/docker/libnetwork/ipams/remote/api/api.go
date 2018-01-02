// Package api defines the data structure to be used in the request/response
// messages between libnetwork and the remote ipam plugin
package api

import "github.com/docker/libnetwork/ipamapi"

// Response is the basic response structure used in all responses
type Response struct {
	Error string
}

// IsSuccess returns whether the plugin response is successful
func (r *Response) IsSuccess() bool {
	return r.Error == ""
}

// GetError returns the error from the response, if any.
func (r *Response) GetError() string {
	return r.Error
}

// GetCapabilityResponse is the response of GetCapability request
type GetCapabilityResponse struct {
	Response
	RequiresMACAddress    bool
	RequiresRequestReplay bool
}

// ToCapability converts the capability response into the internal ipam driver capability structure
func (capRes GetCapabilityResponse) ToCapability() *ipamapi.Capability {
	return &ipamapi.Capability{
		RequiresMACAddress:    capRes.RequiresMACAddress,
		RequiresRequestReplay: capRes.RequiresRequestReplay,
	}
}

// GetAddressSpacesResponse is the response to the ``get default address spaces`` request message
type GetAddressSpacesResponse struct {
	Response
	LocalDefaultAddressSpace  string
	GlobalDefaultAddressSpace string
}

// RequestPoolRequest represents the expected data in a ``request address pool`` request message
type RequestPoolRequest struct {
	AddressSpace string
	Pool         string
	SubPool      string
	Options      map[string]string
	V6           bool
}

// RequestPoolResponse represents the response message to a ``request address pool`` request
type RequestPoolResponse struct {
	Response
	PoolID string
	Pool   string // CIDR format
	Data   map[string]string
}

// ReleasePoolRequest represents the expected data in a ``release address pool`` request message
type ReleasePoolRequest struct {
	PoolID string
}

// ReleasePoolResponse represents the response message to a ``release address pool`` request
type ReleasePoolResponse struct {
	Response
}

// RequestAddressRequest represents the expected data in a ``request address`` request message
type RequestAddressRequest struct {
	PoolID  string
	Address string
	Options map[string]string
}

// RequestAddressResponse represents the expected data in the response message to a ``request address`` request
type RequestAddressResponse struct {
	Response
	Address string // in CIDR format
	Data    map[string]string
}

// ReleaseAddressRequest represents the expected data in a ``release address`` request message
type ReleaseAddressRequest struct {
	PoolID  string
	Address string
}

// ReleaseAddressResponse represents the response message to a ``release address`` request
type ReleaseAddressResponse struct {
	Response
}
