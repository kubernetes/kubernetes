/*
Package api represents all requests and responses suitable for conversation
with a remote driver.
*/
package api

import (
	"net"

	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
)

// Response is the basic response structure used in all responses.
type Response struct {
	Err string
}

// GetError returns the error from the response, if any.
func (r *Response) GetError() string {
	return r.Err
}

// GetCapabilityResponse is the response of GetCapability request
type GetCapabilityResponse struct {
	Response
	Scope             string
	ConnectivityScope string
}

// AllocateNetworkRequest requests allocation of new network by manager
type AllocateNetworkRequest struct {
	// A network ID that remote plugins are expected to store for future
	// reference.
	NetworkID string

	// A free form map->object interface for communication of options.
	Options map[string]string

	// IPAMData contains the address pool information for this network
	IPv4Data, IPv6Data []driverapi.IPAMData
}

// AllocateNetworkResponse is the response to the AllocateNetworkRequest.
type AllocateNetworkResponse struct {
	Response
	// A free form plugin specific string->string object to be sent in
	// CreateNetworkRequest call in the libnetwork agents
	Options map[string]string
}

// FreeNetworkRequest is the request to free allocated network in the manager
type FreeNetworkRequest struct {
	// The ID of the network to be freed.
	NetworkID string
}

// FreeNetworkResponse is the response to a request for freeing a network.
type FreeNetworkResponse struct {
	Response
}

// CreateNetworkRequest requests a new network.
type CreateNetworkRequest struct {
	// A network ID that remote plugins are expected to store for future
	// reference.
	NetworkID string

	// A free form map->object interface for communication of options.
	Options map[string]interface{}

	// IPAMData contains the address pool information for this network
	IPv4Data, IPv6Data []driverapi.IPAMData
}

// CreateNetworkResponse is the response to the CreateNetworkRequest.
type CreateNetworkResponse struct {
	Response
}

// DeleteNetworkRequest is the request to delete an existing network.
type DeleteNetworkRequest struct {
	// The ID of the network to delete.
	NetworkID string
}

// DeleteNetworkResponse is the response to a request for deleting a network.
type DeleteNetworkResponse struct {
	Response
}

// CreateEndpointRequest is the request to create an endpoint within a network.
type CreateEndpointRequest struct {
	// Provided at create time, this will be the network id referenced.
	NetworkID string
	// The ID of the endpoint for later reference.
	EndpointID string
	Interface  *EndpointInterface
	Options    map[string]interface{}
}

// EndpointInterface represents an interface endpoint.
type EndpointInterface struct {
	Address     string
	AddressIPv6 string
	MacAddress  string
}

// CreateEndpointResponse is the response to the CreateEndpoint action.
type CreateEndpointResponse struct {
	Response
	Interface *EndpointInterface
}

// Interface is the representation of a linux interface.
type Interface struct {
	Address     *net.IPNet
	AddressIPv6 *net.IPNet
	MacAddress  net.HardwareAddr
}

// DeleteEndpointRequest describes the API for deleting an endpoint.
type DeleteEndpointRequest struct {
	NetworkID  string
	EndpointID string
}

// DeleteEndpointResponse is the response to the DeleteEndpoint action.
type DeleteEndpointResponse struct {
	Response
}

// EndpointInfoRequest retrieves information about the endpoint from the network driver.
type EndpointInfoRequest struct {
	NetworkID  string
	EndpointID string
}

// EndpointInfoResponse is the response to an EndpointInfoRequest.
type EndpointInfoResponse struct {
	Response
	Value map[string]interface{}
}

// JoinRequest describes the API for joining an endpoint to a sandbox.
type JoinRequest struct {
	NetworkID  string
	EndpointID string
	SandboxKey string
	Options    map[string]interface{}
}

// InterfaceName is the struct represetation of a pair of devices with source
// and destination, for the purposes of putting an endpoint into a container.
type InterfaceName struct {
	SrcName   string
	DstName   string
	DstPrefix string
}

// StaticRoute is the plain JSON representation of a static route.
type StaticRoute struct {
	Destination string
	RouteType   int
	NextHop     string
}

// JoinResponse is the response to a JoinRequest.
type JoinResponse struct {
	Response
	InterfaceName         *InterfaceName
	Gateway               string
	GatewayIPv6           string
	StaticRoutes          []StaticRoute
	DisableGatewayService bool
}

// LeaveRequest describes the API for detaching an endpoint from a sandbox.
type LeaveRequest struct {
	NetworkID  string
	EndpointID string
}

// LeaveResponse is the answer to LeaveRequest.
type LeaveResponse struct {
	Response
}

// ProgramExternalConnectivityRequest describes the API for programming the external connectivity for the given endpoint.
type ProgramExternalConnectivityRequest struct {
	NetworkID  string
	EndpointID string
	Options    map[string]interface{}
}

// ProgramExternalConnectivityResponse is the answer to ProgramExternalConnectivityRequest.
type ProgramExternalConnectivityResponse struct {
	Response
}

// RevokeExternalConnectivityRequest describes the API for revoking the external connectivity for the given endpoint.
type RevokeExternalConnectivityRequest struct {
	NetworkID  string
	EndpointID string
}

// RevokeExternalConnectivityResponse is the answer to RevokeExternalConnectivityRequest.
type RevokeExternalConnectivityResponse struct {
	Response
}

// DiscoveryNotification represents a discovery notification
type DiscoveryNotification struct {
	DiscoveryType discoverapi.DiscoveryType
	DiscoveryData interface{}
}

// DiscoveryResponse is used by libnetwork to log any plugin error processing the discovery notifications
type DiscoveryResponse struct {
	Response
}
