package bridge

import (
	"fmt"
	"net"
)

// ErrConfigExists error is returned when driver already has a config applied.
type ErrConfigExists struct{}

func (ece *ErrConfigExists) Error() string {
	return "configuration already exists, bridge configuration can be applied only once"
}

// Forbidden denotes the type of this error
func (ece *ErrConfigExists) Forbidden() {}

// ErrInvalidDriverConfig error is returned when Bridge Driver is passed an invalid config
type ErrInvalidDriverConfig struct{}

func (eidc *ErrInvalidDriverConfig) Error() string {
	return "Invalid configuration passed to Bridge Driver"
}

// BadRequest denotes the type of this error
func (eidc *ErrInvalidDriverConfig) BadRequest() {}

// ErrInvalidNetworkConfig error is returned when a network is created on a driver without valid config.
type ErrInvalidNetworkConfig struct{}

func (einc *ErrInvalidNetworkConfig) Error() string {
	return "trying to create a network on a driver without valid config"
}

// Forbidden denotes the type of this error
func (einc *ErrInvalidNetworkConfig) Forbidden() {}

// ErrInvalidContainerConfig error is returned when an endpoint create is attempted with an invalid configuration.
type ErrInvalidContainerConfig struct{}

func (eicc *ErrInvalidContainerConfig) Error() string {
	return "Error in joining a container due to invalid configuration"
}

// BadRequest denotes the type of this error
func (eicc *ErrInvalidContainerConfig) BadRequest() {}

// ErrInvalidEndpointConfig error is returned when an endpoint create is attempted with an invalid endpoint configuration.
type ErrInvalidEndpointConfig struct{}

func (eiec *ErrInvalidEndpointConfig) Error() string {
	return "trying to create an endpoint with an invalid endpoint configuration"
}

// BadRequest denotes the type of this error
func (eiec *ErrInvalidEndpointConfig) BadRequest() {}

// ErrNetworkExists error is returned when a network already exists and another network is created.
type ErrNetworkExists struct{}

func (ene *ErrNetworkExists) Error() string {
	return "network already exists, bridge can only have one network"
}

// Forbidden denotes the type of this error
func (ene *ErrNetworkExists) Forbidden() {}

// ErrIfaceName error is returned when a new name could not be generated.
type ErrIfaceName struct{}

func (ein *ErrIfaceName) Error() string {
	return "failed to find name for new interface"
}

// InternalError denotes the type of this error
func (ein *ErrIfaceName) InternalError() {}

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

// ErrInvalidContainerSubnet is returned when the container subnet (FixedCIDR) is not valid.
type ErrInvalidContainerSubnet struct{}

func (eis *ErrInvalidContainerSubnet) Error() string {
	return "container subnet must be a subset of bridge network"
}

// BadRequest denotes the type of this error
func (eis *ErrInvalidContainerSubnet) BadRequest() {}

// ErrInvalidMtu is returned when the user provided MTU is not valid.
type ErrInvalidMtu int

func (eim ErrInvalidMtu) Error() string {
	return fmt.Sprintf("invalid MTU number: %d", int(eim))
}

// BadRequest denotes the type of this error
func (eim ErrInvalidMtu) BadRequest() {}

// ErrInvalidPort is returned when the container or host port specified in the port binding is not valid.
type ErrInvalidPort string

func (ip ErrInvalidPort) Error() string {
	return fmt.Sprintf("invalid transport port: %s", string(ip))
}

// BadRequest denotes the type of this error
func (ip ErrInvalidPort) BadRequest() {}

// ErrUnsupportedAddressType is returned when the specified address type is not supported.
type ErrUnsupportedAddressType string

func (uat ErrUnsupportedAddressType) Error() string {
	return fmt.Sprintf("unsupported address type: %s", string(uat))
}

// BadRequest denotes the type of this error
func (uat ErrUnsupportedAddressType) BadRequest() {}

// ErrInvalidAddressBinding is returned when the host address specified in the port binding is not valid.
type ErrInvalidAddressBinding string

func (iab ErrInvalidAddressBinding) Error() string {
	return fmt.Sprintf("invalid host address in port binding: %s", string(iab))
}

// BadRequest denotes the type of this error
func (iab ErrInvalidAddressBinding) BadRequest() {}

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

// InvalidSandboxIDError is returned when the passed
// sandbox id is not valid.
type InvalidSandboxIDError string

func (isie InvalidSandboxIDError) Error() string {
	return fmt.Sprintf("invalid sandbox id: %s", string(isie))
}

// BadRequest denotes the type of this error
func (isie InvalidSandboxIDError) BadRequest() {}

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

// FixedCIDRv4Error is returned when fixed-cidrv4 configuration
// failed.
type FixedCIDRv4Error struct {
	Net    *net.IPNet
	Subnet *net.IPNet
	Err    error
}

func (fcv4 *FixedCIDRv4Error) Error() string {
	return fmt.Sprintf("setup FixedCIDRv4 failed for subnet %s in %s: %v", fcv4.Subnet, fcv4.Net, fcv4.Err)
}

// InternalError denotes the type of this error
func (fcv4 *FixedCIDRv4Error) InternalError() {}

// FixedCIDRv6Error is returned when fixed-cidrv6 configuration
// failed.
type FixedCIDRv6Error struct {
	Net *net.IPNet
	Err error
}

func (fcv6 *FixedCIDRv6Error) Error() string {
	return fmt.Sprintf("setup FixedCIDRv6 failed for subnet %s in %s: %v", fcv6.Net, fcv6.Net, fcv6.Err)
}

// InternalError denotes the type of this error
func (fcv6 *FixedCIDRv6Error) InternalError() {}

// IPTableCfgError is returned when an unexpected ip tables configuration is entered
type IPTableCfgError string

func (name IPTableCfgError) Error() string {
	return fmt.Sprintf("unexpected request to set IP tables for interface: %s", string(name))
}

// BadRequest denotes the type of this error
func (name IPTableCfgError) BadRequest() {}

// InvalidIPTablesCfgError is returned when an invalid ip tables configuration is entered
type InvalidIPTablesCfgError string

func (action InvalidIPTablesCfgError) Error() string {
	return fmt.Sprintf("Invalid IPTables action '%s'", string(action))
}

// BadRequest denotes the type of this error
func (action InvalidIPTablesCfgError) BadRequest() {}

// IPv4AddrRangeError is returned when a valid IP address range couldn't be found.
type IPv4AddrRangeError string

func (name IPv4AddrRangeError) Error() string {
	return fmt.Sprintf("can't find an address range for interface %q", string(name))
}

// BadRequest denotes the type of this error
func (name IPv4AddrRangeError) BadRequest() {}

// IPv4AddrAddError is returned when IPv4 address could not be added to the bridge.
type IPv4AddrAddError struct {
	IP  *net.IPNet
	Err error
}

func (ipv4 *IPv4AddrAddError) Error() string {
	return fmt.Sprintf("failed to add IPv4 address %s to bridge: %v", ipv4.IP, ipv4.Err)
}

// InternalError denotes the type of this error
func (ipv4 *IPv4AddrAddError) InternalError() {}

// IPv6AddrAddError is returned when IPv6 address could not be added to the bridge.
type IPv6AddrAddError struct {
	IP  *net.IPNet
	Err error
}

func (ipv6 *IPv6AddrAddError) Error() string {
	return fmt.Sprintf("failed to add IPv6 address %s to bridge: %v", ipv6.IP, ipv6.Err)
}

// InternalError denotes the type of this error
func (ipv6 *IPv6AddrAddError) InternalError() {}

// IPv4AddrNoMatchError is returned when the bridge's IPv4 address does not match configured.
type IPv4AddrNoMatchError struct {
	IP    net.IP
	CfgIP net.IP
}

func (ipv4 *IPv4AddrNoMatchError) Error() string {
	return fmt.Sprintf("bridge IPv4 (%s) does not match requested configuration %s", ipv4.IP, ipv4.CfgIP)
}

// BadRequest denotes the type of this error
func (ipv4 *IPv4AddrNoMatchError) BadRequest() {}

// IPv6AddrNoMatchError is returned when the bridge's IPv6 address does not match configured.
type IPv6AddrNoMatchError net.IPNet

func (ipv6 *IPv6AddrNoMatchError) Error() string {
	return fmt.Sprintf("bridge IPv6 addresses do not match the expected bridge configuration %s", (*net.IPNet)(ipv6).String())
}

// BadRequest denotes the type of this error
func (ipv6 *IPv6AddrNoMatchError) BadRequest() {}

// InvalidLinkIPAddrError is returned when a link is configured to a container with an invalid ip address
type InvalidLinkIPAddrError string

func (address InvalidLinkIPAddrError) Error() string {
	return fmt.Sprintf("Cannot link to a container with Invalid IP Address '%s'", string(address))
}

// BadRequest denotes the type of this error
func (address InvalidLinkIPAddrError) BadRequest() {}
