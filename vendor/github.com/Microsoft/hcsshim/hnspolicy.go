package hcsshim

// Type of Request Support in ModifySystem
type PolicyType string

// RequestType const
const (
	Nat                  PolicyType = "NAT"
	ACL                  PolicyType = "ACL"
	PA                   PolicyType = "PA"
	VLAN                 PolicyType = "VLAN"
	VSID                 PolicyType = "VSID"
	VNet                 PolicyType = "VNET"
	L2Driver             PolicyType = "L2Driver"
	Isolation            PolicyType = "Isolation"
	QOS                  PolicyType = "QOS"
	OutboundNat          PolicyType = "OutBoundNAT"
	ExternalLoadBalancer PolicyType = "ELB"
	Route                PolicyType = "ROUTE"
)

type NatPolicy struct {
	Type         PolicyType `json:"Type"`
	Protocol     string
	InternalPort uint16
	ExternalPort uint16
}

type QosPolicy struct {
	Type                            PolicyType `json:"Type"`
	MaximumOutgoingBandwidthInBytes uint64
}

type IsolationPolicy struct {
	Type               PolicyType `json:"Type"`
	VLAN               uint
	VSID               uint
	InDefaultIsolation bool
}

type VlanPolicy struct {
	Type PolicyType `json:"Type"`
	VLAN uint
}

type VsidPolicy struct {
	Type PolicyType `json:"Type"`
	VSID uint
}

type PaPolicy struct {
	Type PolicyType `json:"Type"`
	PA   string     `json:"PA"`
}

type OutboundNatPolicy struct {
	Policy
	VIP        string   `json:"VIP,omitempty"`
	Exceptions []string `json:"ExceptionList,omitempty"`
}

type ActionType string
type DirectionType string
type RuleType string

const (
	Allow ActionType = "Allow"
	Block ActionType = "Block"

	In  DirectionType = "In"
	Out DirectionType = "Out"

	Host   RuleType = "Host"
	Switch RuleType = "Switch"
)

type ACLPolicy struct {
	Type          PolicyType `json:"Type"`
	Protocol      uint16
	InternalPort  uint16
	Action        ActionType
	Direction     DirectionType
	LocalAddress  string
	RemoteAddress string
	LocalPort     uint16
	RemotePort    uint16
	RuleType      RuleType `json:"RuleType,omitempty"`

	Priority    uint16
	ServiceName string
}

type Policy struct {
	Type PolicyType `json:"Type"`
}
