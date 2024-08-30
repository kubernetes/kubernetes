//go:build windows

package hcn

import (
	"encoding/json"
)

// EndpointPolicyType are the potential Policies that apply to Endpoints.
type EndpointPolicyType string

// EndpointPolicyType const
const (
	PortMapping   EndpointPolicyType = "PortMapping"
	ACL           EndpointPolicyType = "ACL"
	QOS           EndpointPolicyType = "QOS"
	L2Driver      EndpointPolicyType = "L2Driver"
	OutBoundNAT   EndpointPolicyType = "OutBoundNAT"
	SDNRoute      EndpointPolicyType = "SDNRoute"
	L4Proxy       EndpointPolicyType = "L4Proxy"
	L4WFPPROXY    EndpointPolicyType = "L4WFPPROXY"
	PortName      EndpointPolicyType = "PortName"
	EncapOverhead EndpointPolicyType = "EncapOverhead"
	IOV           EndpointPolicyType = "Iov"
	// Endpoint and Network have InterfaceConstraint and ProviderAddress
	NetworkProviderAddress     EndpointPolicyType = "ProviderAddress"
	NetworkInterfaceConstraint EndpointPolicyType = "InterfaceConstraint"
	TierAcl                    EndpointPolicyType = "TierAcl"
)

// EndpointPolicy is a collection of Policy settings for an Endpoint.
type EndpointPolicy struct {
	Type     EndpointPolicyType `json:""`
	Settings json.RawMessage    `json:",omitempty"`
}

// NetworkPolicyType are the potential Policies that apply to Networks.
type NetworkPolicyType string

// NetworkPolicyType const
const (
	SourceMacAddress    NetworkPolicyType = "SourceMacAddress"
	NetAdapterName      NetworkPolicyType = "NetAdapterName"
	VSwitchExtension    NetworkPolicyType = "VSwitchExtension"
	DrMacAddress        NetworkPolicyType = "DrMacAddress"
	AutomaticDNS        NetworkPolicyType = "AutomaticDNS"
	InterfaceConstraint NetworkPolicyType = "InterfaceConstraint"
	ProviderAddress     NetworkPolicyType = "ProviderAddress"
	RemoteSubnetRoute   NetworkPolicyType = "RemoteSubnetRoute"
	VxlanPort           NetworkPolicyType = "VxlanPort"
	HostRoute           NetworkPolicyType = "HostRoute"
	SetPolicy           NetworkPolicyType = "SetPolicy"
	NetworkL4Proxy      NetworkPolicyType = "L4Proxy"
	LayerConstraint     NetworkPolicyType = "LayerConstraint"
	NetworkACL          NetworkPolicyType = "NetworkACL"
)

// NetworkPolicy is a collection of Policy settings for a Network.
type NetworkPolicy struct {
	Type     NetworkPolicyType `json:""`
	Settings json.RawMessage   `json:",omitempty"`
}

// SubnetPolicyType are the potential Policies that apply to Subnets.
type SubnetPolicyType string

// SubnetPolicyType const
const (
	VLAN SubnetPolicyType = "VLAN"
	VSID SubnetPolicyType = "VSID"
)

// SubnetPolicy is a collection of Policy settings for a Subnet.
type SubnetPolicy struct {
	Type     SubnetPolicyType `json:""`
	Settings json.RawMessage  `json:",omitempty"`
}

// NatFlags are flags for portmappings.
type NatFlags uint32

const (
	NatFlagsNone NatFlags = iota
	NatFlagsLocalRoutedVip
	NatFlagsIPv6
)

/// Endpoint Policy objects

// PortMappingPolicySetting defines Port Mapping (NAT)
type PortMappingPolicySetting struct {
	Protocol     uint32   `json:",omitempty"` // EX: TCP = 6, UDP = 17
	InternalPort uint16   `json:",omitempty"`
	ExternalPort uint16   `json:",omitempty"`
	VIP          string   `json:",omitempty"`
	Flags        NatFlags `json:",omitempty"`
}

// ActionType associated with ACLs. Value is either Allow or Block.
type ActionType string

// DirectionType associated with ACLs. Value is either In or Out.
type DirectionType string

// RuleType associated with ACLs. Value is either Host (WFP) or Switch (VFP).
type RuleType string

const (
	// Allow traffic
	ActionTypeAllow ActionType = "Allow"
	// Block traffic
	ActionTypeBlock ActionType = "Block"
	// Pass traffic
	ActionTypePass ActionType = "Pass"

	// In is traffic coming to the Endpoint
	DirectionTypeIn DirectionType = "In"
	// Out is traffic leaving the Endpoint
	DirectionTypeOut DirectionType = "Out"

	// Host creates WFP (Windows Firewall) rules
	RuleTypeHost RuleType = "Host"
	// Switch creates VFP (Virtual Filter Platform) rules
	RuleTypeSwitch RuleType = "Switch"
)

// AclPolicySetting creates firewall rules on an endpoint
type AclPolicySetting struct {
	Protocols       string        `json:",omitempty"` // EX: 6 (TCP), 17 (UDP), 1 (ICMPv4), 58 (ICMPv6), 2 (IGMP)
	Action          ActionType    `json:","`
	Direction       DirectionType `json:","`
	LocalAddresses  string        `json:",omitempty"`
	RemoteAddresses string        `json:",omitempty"`
	LocalPorts      string        `json:",omitempty"`
	RemotePorts     string        `json:",omitempty"`
	RuleType        RuleType      `json:",omitempty"`
	Priority        uint16        `json:",omitempty"`
}

// QosPolicySetting sets Quality of Service bandwidth caps on an Endpoint.
type QosPolicySetting struct {
	MaximumOutgoingBandwidthInBytes uint64
}

// OutboundNatPolicySetting sets outbound Network Address Translation on an Endpoint.
type OutboundNatPolicySetting struct {
	VirtualIP        string   `json:",omitempty"`
	Exceptions       []string `json:",omitempty"`
	Destinations     []string `json:",omitempty"`
	Flags            NatFlags `json:",omitempty"`
	MaxPortPoolUsage uint16   `json:",omitempty"`
}

// SDNRoutePolicySetting sets SDN Route on an Endpoint.
type SDNRoutePolicySetting struct {
	DestinationPrefix string `json:",omitempty"`
	NextHop           string `json:",omitempty"`
	NeedEncap         bool   `json:",omitempty"`
}

// NetworkACLPolicySetting creates ACL rules on a network
type NetworkACLPolicySetting struct {
	Protocols       string        `json:",omitempty"` // EX: 6 (TCP), 17 (UDP), 1 (ICMPv4), 58 (ICMPv6), 2 (IGMP)
	Action          ActionType    `json:","`
	Direction       DirectionType `json:","`
	LocalAddresses  string        `json:",omitempty"`
	RemoteAddresses string        `json:",omitempty"`
	LocalPorts      string        `json:",omitempty"`
	RemotePorts     string        `json:",omitempty"`
	RuleType        RuleType      `json:",omitempty"`
	Priority        uint16        `json:",omitempty"`
}

// FiveTuple is nested in L4ProxyPolicySetting  for WFP support.
type FiveTuple struct {
	Protocols       string `json:",omitempty"`
	LocalAddresses  string `json:",omitempty"`
	RemoteAddresses string `json:",omitempty"`
	LocalPorts      string `json:",omitempty"`
	RemotePorts     string `json:",omitempty"`
	Priority        uint16 `json:",omitempty"`
}

// ProxyExceptions exempts traffic to IpAddresses and Ports
type ProxyExceptions struct {
	IpAddressExceptions []string `json:",omitempty"`
	PortExceptions      []string `json:",omitempty"`
}

// L4WfpProxyPolicySetting sets Layer-4 Proxy on an endpoint.
type L4WfpProxyPolicySetting struct {
	InboundProxyPort   string          `json:",omitempty"`
	OutboundProxyPort  string          `json:",omitempty"`
	FilterTuple        FiveTuple       `json:",omitempty"`
	UserSID            string          `json:",omitempty"`
	InboundExceptions  ProxyExceptions `json:",omitempty"`
	OutboundExceptions ProxyExceptions `json:",omitempty"`
}

// PortnameEndpointPolicySetting sets the port name for an endpoint.
type PortnameEndpointPolicySetting struct {
	Name string `json:",omitempty"`
}

// EncapOverheadEndpointPolicySetting sets the encap overhead for an endpoint.
type EncapOverheadEndpointPolicySetting struct {
	Overhead uint16 `json:",omitempty"`
}

// IovPolicySetting sets the Iov settings for an endpoint.
type IovPolicySetting struct {
	IovOffloadWeight    uint32 `json:",omitempty"`
	QueuePairsRequested uint32 `json:",omitempty"`
	InterruptModeration uint32 `json:",omitempty"`
}

/// Endpoint and Network Policy objects

// ProviderAddressEndpointPolicySetting sets the PA for an endpoint.
type ProviderAddressEndpointPolicySetting struct {
	ProviderAddress string `json:",omitempty"`
}

// InterfaceConstraintPolicySetting limits an Endpoint or Network to a specific Nic.
type InterfaceConstraintPolicySetting struct {
	InterfaceGuid        string `json:",omitempty"`
	InterfaceLuid        uint64 `json:",omitempty"`
	InterfaceIndex       uint32 `json:",omitempty"`
	InterfaceMediaType   uint32 `json:",omitempty"`
	InterfaceAlias       string `json:",omitempty"`
	InterfaceDescription string `json:",omitempty"`
}

/// Network Policy objects

// SourceMacAddressNetworkPolicySetting sets source MAC for a network.
type SourceMacAddressNetworkPolicySetting struct {
	SourceMacAddress string `json:",omitempty"`
}

// NetAdapterNameNetworkPolicySetting sets network adapter of a network.
type NetAdapterNameNetworkPolicySetting struct {
	NetworkAdapterName string `json:",omitempty"`
}

// VSwitchExtensionNetworkPolicySetting enables/disabled VSwitch extensions for a network.
type VSwitchExtensionNetworkPolicySetting struct {
	ExtensionID string `json:",omitempty"`
	Enable      bool   `json:",omitempty"`
}

// DrMacAddressNetworkPolicySetting sets the DR MAC for a network.
type DrMacAddressNetworkPolicySetting struct {
	Address string `json:",omitempty"`
}

// AutomaticDNSNetworkPolicySetting enables/disables automatic DNS on a network.
type AutomaticDNSNetworkPolicySetting struct {
	Enable bool `json:",omitempty"`
}

type LayerConstraintNetworkPolicySetting struct {
	LayerId string `json:",omitempty"`
}

/// Subnet Policy objects

// VlanPolicySetting isolates a subnet with VLAN tagging.
type VlanPolicySetting struct {
	IsolationId uint32 `json:","`
}

// VsidPolicySetting isolates a subnet with VSID tagging.
type VsidPolicySetting struct {
	IsolationId uint32 `json:","`
}

// RemoteSubnetRoutePolicySetting creates remote subnet route rules on a network
type RemoteSubnetRoutePolicySetting struct {
	DestinationPrefix           string
	IsolationId                 uint16
	ProviderAddress             string
	DistributedRouterMacAddress string
}

// SetPolicyTypes associated with SetPolicy. Value is IPSET.
type SetPolicyType string

const (
	SetPolicyTypeIpSet       SetPolicyType = "IPSET"
	SetPolicyTypeNestedIpSet SetPolicyType = "NESTEDIPSET"
)

// SetPolicySetting creates IPSets on network
type SetPolicySetting struct {
	Id     string
	Name   string
	Type   SetPolicyType `json:"PolicyType"`
	Values string
}

// VxlanPortPolicySetting allows configuring the VXLAN TCP port
type VxlanPortPolicySetting struct {
	Port uint16
}

// ProtocolType associated with L4ProxyPolicy
type ProtocolType uint32

const (
	ProtocolTypeUnknown ProtocolType = 0
	ProtocolTypeICMPv4  ProtocolType = 1
	ProtocolTypeIGMP    ProtocolType = 2
	ProtocolTypeTCP     ProtocolType = 6
	ProtocolTypeUDP     ProtocolType = 17
	ProtocolTypeICMPv6  ProtocolType = 58
)

// L4ProxyPolicySetting applies proxy policy on network/endpoint
type L4ProxyPolicySetting struct {
	IP          string       `json:",omitempty"`
	Port        string       `json:",omitempty"`
	Protocol    ProtocolType `json:",omitempty"`
	Exceptions  []string     `json:",omitempty"`
	Destination string
	OutboundNAT bool `json:",omitempty"`
}

// TierAclRule represents an ACL within TierAclPolicySetting
type TierAclRule struct {
	Id                string     `json:",omitempty"`
	Protocols         string     `json:",omitempty"`
	TierAclRuleAction ActionType `json:","`
	LocalAddresses    string     `json:",omitempty"`
	RemoteAddresses   string     `json:",omitempty"`
	LocalPorts        string     `json:",omitempty"`
	RemotePorts       string     `json:",omitempty"`
	Priority          uint16     `json:",omitempty"`
}

// TierAclPolicySetting represents a Tier containing ACLs
type TierAclPolicySetting struct {
	Name         string        `json:","`
	Direction    DirectionType `json:","`
	Order        uint16        `json:""`
	TierAclRules []TierAclRule `json:",omitempty"`
}
