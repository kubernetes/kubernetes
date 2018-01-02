// Package osl describes structures and interfaces which abstract os entities
package osl

import (
	"net"

	"github.com/docker/libnetwork/types"
)

// Sandbox represents a network sandbox, identified by a specific key.  It
// holds a list of Interfaces, routes etc, and more can be added dynamically.
type Sandbox interface {
	// The path where the network namespace is mounted.
	Key() string

	// Add an existing Interface to this sandbox. The operation will rename
	// from the Interface SrcName to DstName as it moves, and reconfigure the
	// interface according to the specified settings. The caller is expected
	// to only provide a prefix for DstName. The AddInterface api will auto-generate
	// an appropriate suffix for the DstName to disambiguate.
	AddInterface(SrcName string, DstPrefix string, options ...IfaceOption) error

	// Set default IPv4 gateway for the sandbox
	SetGateway(gw net.IP) error

	// Set default IPv6 gateway for the sandbox
	SetGatewayIPv6(gw net.IP) error

	// Unset the previously set default IPv4 gateway in the sandbox
	UnsetGateway() error

	// Unset the previously set default IPv6 gateway in the sandbox
	UnsetGatewayIPv6() error

	// AddLoopbackAliasIP adds the passed IP address to the sandbox loopback interface
	AddLoopbackAliasIP(ip *net.IPNet) error

	// RemoveLoopbackAliasIP removes the passed IP address from the sandbox loopback interface
	RemoveLoopbackAliasIP(ip *net.IPNet) error

	// Add a static route to the sandbox.
	AddStaticRoute(*types.StaticRoute) error

	// Remove a static route from the sandbox.
	RemoveStaticRoute(*types.StaticRoute) error

	// AddNeighbor adds a neighbor entry into the sandbox.
	AddNeighbor(dstIP net.IP, dstMac net.HardwareAddr, force bool, option ...NeighOption) error

	// DeleteNeighbor deletes neighbor entry from the sandbox.
	DeleteNeighbor(dstIP net.IP, dstMac net.HardwareAddr, osDelete bool) error

	// Returns an interface with methods to set neighbor options.
	NeighborOptions() NeighborOptionSetter

	// Returns an interface with methods to set interface options.
	InterfaceOptions() IfaceOptionSetter

	//Invoke
	InvokeFunc(func()) error

	// Returns an interface with methods to get sandbox state.
	Info() Info

	// Destroy the sandbox
	Destroy() error

	// restore sandbox
	Restore(ifsopt map[string][]IfaceOption, routes []*types.StaticRoute, gw net.IP, gw6 net.IP) error
}

// NeighborOptionSetter interface defines the option setter methods for interface options
type NeighborOptionSetter interface {
	// LinkName returns an option setter to set the srcName of the link that should
	// be used in the neighbor entry
	LinkName(string) NeighOption

	// Family returns an option setter to set the address family for the neighbor
	// entry. eg. AF_BRIDGE
	Family(int) NeighOption
}

// IfaceOptionSetter interface defines the option setter methods for interface options.
type IfaceOptionSetter interface {
	// Bridge returns an option setter to set if the interface is a bridge.
	Bridge(bool) IfaceOption

	// MacAddress returns an option setter to set the MAC address.
	MacAddress(net.HardwareAddr) IfaceOption

	// Address returns an option setter to set IPv4 address.
	Address(*net.IPNet) IfaceOption

	// Address returns an option setter to set IPv6 address.
	AddressIPv6(*net.IPNet) IfaceOption

	// LinkLocalAddresses returns an option setter to set the link-local IP addresses.
	LinkLocalAddresses([]*net.IPNet) IfaceOption

	// Master returns an option setter to set the master interface if any for this
	// interface. The master interface name should refer to the srcname of a
	// previously added interface of type bridge.
	Master(string) IfaceOption

	// Address returns an option setter to set interface routes.
	Routes([]*net.IPNet) IfaceOption
}

// Info represents all possible information that
// the driver wants to place in the sandbox which includes
// interfaces, routes and gateway
type Info interface {
	// The collection of Interface previously added with the AddInterface
	// method. Note that this doesn't include network interfaces added in any
	// other way (such as the default loopback interface which is automatically
	// created on creation of a sandbox).
	Interfaces() []Interface

	// IPv4 gateway for the sandbox.
	Gateway() net.IP

	// IPv6 gateway for the sandbox.
	GatewayIPv6() net.IP

	// Additional static routes for the sandbox.  (Note that directly
	// connected routes are stored on the particular interface they refer to.)
	StaticRoutes() []*types.StaticRoute

	// TODO: Add ip tables etc.
}

// Interface represents the settings and identity of a network device. It is
// used as a return type for Network.Link, and it is common practice for the
// caller to use this information when moving interface SrcName from host
// namespace to DstName in a different net namespace with the appropriate
// network settings.
type Interface interface {
	// The name of the interface in the origin network namespace.
	SrcName() string

	// The name that will be assigned to the interface once moves inside a
	// network namespace. When the caller passes in a DstName, it is only
	// expected to pass a prefix. The name will modified with an appropriately
	// auto-generated suffix.
	DstName() string

	// IPv4 address for the interface.
	Address() *net.IPNet

	// IPv6 address for the interface.
	AddressIPv6() *net.IPNet

	// LinkLocalAddresses returns the link-local IP addresses assigned to the interface.
	LinkLocalAddresses() []*net.IPNet

	// IP routes for the interface.
	Routes() []*net.IPNet

	// Bridge returns true if the interface is a bridge
	Bridge() bool

	// Master returns the srcname of the master interface for this interface.
	Master() string

	// Remove an interface from the sandbox by renaming to original name
	// and moving it out of the sandbox.
	Remove() error

	// Statistics returns the statistics for this interface
	Statistics() (*types.InterfaceStatistics, error)
}
