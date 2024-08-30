package hcnshim

import (
	"github.com/Microsoft/hcnshim/internal/hns"
)

// Type of Request Support in ModifySystem
type PolicyType = hns.PolicyType

// RequestType const
const (
	Nat                  = hns.Nat
	ACL                  = hns.ACL
	PA                   = hns.PA
	VLAN                 = hns.VLAN
	VSID                 = hns.VSID
	VNet                 = hns.VNet
	L2Driver             = hns.L2Driver
	Isolation            = hns.Isolation
	QOS                  = hns.QOS
	OutboundNat          = hns.OutboundNat
	ExternalLoadBalancer = hns.ExternalLoadBalancer
	Route                = hns.Route
	Proxy                = hns.Proxy
)

type ProxyPolicy = hns.ProxyPolicy

type NatPolicy = hns.NatPolicy

type QosPolicy = hns.QosPolicy

type IsolationPolicy = hns.IsolationPolicy

type VlanPolicy = hns.VlanPolicy

type VsidPolicy = hns.VsidPolicy

type PaPolicy = hns.PaPolicy

type OutboundNatPolicy = hns.OutboundNatPolicy

type ActionType = hns.ActionType
type DirectionType = hns.DirectionType
type RuleType = hns.RuleType

const (
	Allow = hns.Allow
	Block = hns.Block

	In  = hns.In
	Out = hns.Out

	Host   = hns.Host
	Switch = hns.Switch
)

type ACLPolicy = hns.ACLPolicy

type Policy = hns.Policy
