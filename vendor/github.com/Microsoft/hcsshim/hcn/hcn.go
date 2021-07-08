// Package hcn is a shim for the Host Compute Networking (HCN) service, which manages networking for Windows Server
// containers and Hyper-V containers. Previous to RS5, HCN was referred to as Host Networking Service (HNS).
package hcn

import (
	"fmt"
	"syscall"

	"github.com/Microsoft/go-winio/pkg/guid"
)

//go:generate go run ../mksyscall_windows.go -output zsyscall_windows.go hcn.go

/// HNS V1 API

//sys SetCurrentThreadCompartmentId(compartmentId uint32) (hr error) = iphlpapi.SetCurrentThreadCompartmentId
//sys _hnsCall(method string, path string, object string, response **uint16) (hr error) = vmcompute.HNSCall?

/// HCN V2 API

// Network
//sys hcnEnumerateNetworks(query string, networks **uint16, result **uint16) (hr error) = computenetwork.HcnEnumerateNetworks?
//sys hcnCreateNetwork(id *_guid, settings string, network *hcnNetwork, result **uint16) (hr error) = computenetwork.HcnCreateNetwork?
//sys hcnOpenNetwork(id *_guid, network *hcnNetwork, result **uint16) (hr error) = computenetwork.HcnOpenNetwork?
//sys hcnModifyNetwork(network hcnNetwork, settings string, result **uint16) (hr error) = computenetwork.HcnModifyNetwork?
//sys hcnQueryNetworkProperties(network hcnNetwork, query string, properties **uint16, result **uint16) (hr error) = computenetwork.HcnQueryNetworkProperties?
//sys hcnDeleteNetwork(id *_guid, result **uint16) (hr error) = computenetwork.HcnDeleteNetwork?
//sys hcnCloseNetwork(network hcnNetwork) (hr error) = computenetwork.HcnCloseNetwork?

// Endpoint
//sys hcnEnumerateEndpoints(query string, endpoints **uint16, result **uint16) (hr error) = computenetwork.HcnEnumerateEndpoints?
//sys hcnCreateEndpoint(network hcnNetwork, id *_guid, settings string, endpoint *hcnEndpoint, result **uint16) (hr error) = computenetwork.HcnCreateEndpoint?
//sys hcnOpenEndpoint(id *_guid, endpoint *hcnEndpoint, result **uint16) (hr error) = computenetwork.HcnOpenEndpoint?
//sys hcnModifyEndpoint(endpoint hcnEndpoint, settings string, result **uint16) (hr error) = computenetwork.HcnModifyEndpoint?
//sys hcnQueryEndpointProperties(endpoint hcnEndpoint, query string, properties **uint16, result **uint16) (hr error) = computenetwork.HcnQueryEndpointProperties?
//sys hcnDeleteEndpoint(id *_guid, result **uint16) (hr error) = computenetwork.HcnDeleteEndpoint?
//sys hcnCloseEndpoint(endpoint hcnEndpoint) (hr error) = computenetwork.HcnCloseEndpoint?

// Namespace
//sys hcnEnumerateNamespaces(query string, namespaces **uint16, result **uint16) (hr error) = computenetwork.HcnEnumerateNamespaces?
//sys hcnCreateNamespace(id *_guid, settings string, namespace *hcnNamespace, result **uint16) (hr error) = computenetwork.HcnCreateNamespace?
//sys hcnOpenNamespace(id *_guid, namespace *hcnNamespace, result **uint16) (hr error) = computenetwork.HcnOpenNamespace?
//sys hcnModifyNamespace(namespace hcnNamespace, settings string, result **uint16) (hr error) = computenetwork.HcnModifyNamespace?
//sys hcnQueryNamespaceProperties(namespace hcnNamespace, query string, properties **uint16, result **uint16) (hr error) = computenetwork.HcnQueryNamespaceProperties?
//sys hcnDeleteNamespace(id *_guid, result **uint16) (hr error) = computenetwork.HcnDeleteNamespace?
//sys hcnCloseNamespace(namespace hcnNamespace) (hr error) = computenetwork.HcnCloseNamespace?

// LoadBalancer
//sys hcnEnumerateLoadBalancers(query string, loadBalancers **uint16, result **uint16) (hr error) = computenetwork.HcnEnumerateLoadBalancers?
//sys hcnCreateLoadBalancer(id *_guid, settings string, loadBalancer *hcnLoadBalancer, result **uint16) (hr error) = computenetwork.HcnCreateLoadBalancer?
//sys hcnOpenLoadBalancer(id *_guid, loadBalancer *hcnLoadBalancer, result **uint16) (hr error) = computenetwork.HcnOpenLoadBalancer?
//sys hcnModifyLoadBalancer(loadBalancer hcnLoadBalancer, settings string, result **uint16) (hr error) = computenetwork.HcnModifyLoadBalancer?
//sys hcnQueryLoadBalancerProperties(loadBalancer hcnLoadBalancer, query string, properties **uint16, result **uint16) (hr error) = computenetwork.HcnQueryLoadBalancerProperties?
//sys hcnDeleteLoadBalancer(id *_guid, result **uint16) (hr error) = computenetwork.HcnDeleteLoadBalancer?
//sys hcnCloseLoadBalancer(loadBalancer hcnLoadBalancer) (hr error) = computenetwork.HcnCloseLoadBalancer?

// SDN Routes
//sys hcnEnumerateRoutes(query string, routes **uint16, result **uint16) (hr error) = computenetwork.HcnEnumerateSdnRoutes?
//sys hcnCreateRoute(id *_guid, settings string, route *hcnRoute, result **uint16) (hr error) = computenetwork.HcnCreateSdnRoute?
//sys hcnOpenRoute(id *_guid, route *hcnRoute, result **uint16) (hr error) = computenetwork.HcnOpenSdnRoute?
//sys hcnModifyRoute(route hcnRoute, settings string, result **uint16) (hr error) = computenetwork.HcnModifySdnRoute?
//sys hcnQueryRouteProperties(route hcnRoute, query string, properties **uint16, result **uint16) (hr error) = computenetwork.HcnQuerySdnRouteProperties?
//sys hcnDeleteRoute(id *_guid, result **uint16) (hr error) = computenetwork.HcnDeleteSdnRoute?
//sys hcnCloseRoute(route hcnRoute) (hr error) = computenetwork.HcnCloseSdnRoute?

type _guid = guid.GUID

type hcnNetwork syscall.Handle
type hcnEndpoint syscall.Handle
type hcnNamespace syscall.Handle
type hcnLoadBalancer syscall.Handle
type hcnRoute syscall.Handle

// SchemaVersion for HCN Objects/Queries.
type SchemaVersion = Version // hcnglobals.go

// HostComputeQueryFlags are passed in to a HostComputeQuery to determine which
// properties of an object are returned.
type HostComputeQueryFlags uint32

var (
	// HostComputeQueryFlagsNone returns an object with the standard properties.
	HostComputeQueryFlagsNone HostComputeQueryFlags
	// HostComputeQueryFlagsDetailed returns an object with all properties.
	HostComputeQueryFlagsDetailed HostComputeQueryFlags = 1
)

// HostComputeQuery is the format for HCN queries.
type HostComputeQuery struct {
	SchemaVersion SchemaVersion         `json:""`
	Flags         HostComputeQueryFlags `json:",omitempty"`
	Filter        string                `json:",omitempty"`
}

type ExtraParams struct {
	Resources        interface{} `json:",omitempty"`
	SharedContainers interface{} `json:",omitempty"`
	LayeredOn        string      `json:",omitempty"`
	SwitchGuid       string      `json:",omitempty"`
	UtilityVM        string      `json:",omitempty"`
	VirtualMachine   string      `json:",omitempty"`
}

type Health struct {
	Data  interface{} `json:",omitempty"`
	Extra ExtraParams `json:",omitempty"`
}

// defaultQuery generates HCN Query.
// Passed into get/enumerate calls to filter results.
func defaultQuery() HostComputeQuery {
	query := HostComputeQuery{
		SchemaVersion: SchemaVersion{
			Major: 2,
			Minor: 0,
		},
		Flags: HostComputeQueryFlagsNone,
	}
	return query
}

// PlatformDoesNotSupportError happens when users are attempting to use a newer shim on an older OS
func platformDoesNotSupportError(featureName string) error {
	return fmt.Errorf("Platform does not support feature %s", featureName)
}

// V2ApiSupported returns an error if the HCN version does not support the V2 Apis.
func V2ApiSupported() error {
	supported := GetSupportedFeatures()
	if supported.Api.V2 {
		return nil
	}
	return platformDoesNotSupportError("V2 Api/Schema")
}

func V2SchemaVersion() SchemaVersion {
	return SchemaVersion{
		Major: 2,
		Minor: 0,
	}
}

// RemoteSubnetSupported returns an error if the HCN version does not support Remote Subnet policies.
func RemoteSubnetSupported() error {
	supported := GetSupportedFeatures()
	if supported.RemoteSubnet {
		return nil
	}
	return platformDoesNotSupportError("Remote Subnet")
}

// HostRouteSupported returns an error if the HCN version does not support Host Route policies.
func HostRouteSupported() error {
	supported := GetSupportedFeatures()
	if supported.HostRoute {
		return nil
	}
	return platformDoesNotSupportError("Host Route")
}

// DSRSupported returns an error if the HCN version does not support Direct Server Return.
func DSRSupported() error {
	supported := GetSupportedFeatures()
	if supported.DSR {
		return nil
	}
	return platformDoesNotSupportError("Direct Server Return (DSR)")
}

// Slash32EndpointPrefixesSupported returns an error if the HCN version does not support configuring endpoints with /32 prefixes.
func Slash32EndpointPrefixesSupported() error {
	supported := GetSupportedFeatures()
	if supported.Slash32EndpointPrefixes {
		return nil
	}
	return platformDoesNotSupportError("Slash 32 Endpoint prefixes")
}

// AclSupportForProtocol252Supported returns an error if the HCN version does not support HNS ACL Policies to support protocol 252 for VXLAN.
func AclSupportForProtocol252Supported() error {
	supported := GetSupportedFeatures()
	if supported.AclSupportForProtocol252 {
		return nil
	}
	return platformDoesNotSupportError("HNS ACL Policies to support protocol 252 for VXLAN")
}

// SessionAffinitySupported returns an error if the HCN version does not support Session Affinity.
func SessionAffinitySupported() error {
	supported := GetSupportedFeatures()
	if supported.SessionAffinity {
		return nil
	}
	return platformDoesNotSupportError("Session Affinity")
}

// IPv6DualStackSupported returns an error if the HCN version does not support IPv6DualStack.
func IPv6DualStackSupported() error {
	supported := GetSupportedFeatures()
	if supported.IPv6DualStack {
		return nil
	}
	return platformDoesNotSupportError("IPv6 DualStack")
}

//L4proxySupported returns an error if the HCN verison does not support L4Proxy
func L4proxyPolicySupported() error {
	supported := GetSupportedFeatures()
	if supported.L4Proxy {
		return nil
	}
	return platformDoesNotSupportError("L4ProxyPolicy")
}

// L4WfpProxySupported returns an error if the HCN verison does not support L4WfpProxy
func L4WfpProxyPolicySupported() error {
	supported := GetSupportedFeatures()
	if supported.L4WfpProxy {
		return nil
	}
	return platformDoesNotSupportError("L4WfpProxyPolicy")
}

// SetPolicySupported returns an error if the HCN version does not support SetPolicy.
func SetPolicySupported() error {
	supported := GetSupportedFeatures()
	if supported.SetPolicy {
		return nil
	}
	return platformDoesNotSupportError("SetPolicy")
}

// VxlanPortSupported returns an error if the HCN version does not support configuring the VXLAN TCP port.
func VxlanPortSupported() error {
	supported := GetSupportedFeatures()
	if supported.VxlanPort {
		return nil
	}
	return platformDoesNotSupportError("VXLAN port configuration")
}

// RequestType are the different operations performed to settings.
// Used to update the settings of Endpoint/Namespace objects.
type RequestType string

var (
	// RequestTypeAdd adds the provided settings object.
	RequestTypeAdd RequestType = "Add"
	// RequestTypeRemove removes the provided settings object.
	RequestTypeRemove RequestType = "Remove"
	// RequestTypeUpdate replaces settings with the ones provided.
	RequestTypeUpdate RequestType = "Update"
	// RequestTypeRefresh refreshes the settings provided.
	RequestTypeRefresh RequestType = "Refresh"
)
