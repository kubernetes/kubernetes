/*
Copyright 2023 Red Hat, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package knftables

import (
	"io"
	"time"
)

const (
	// Maximum length of a table, chain, set, etc, name
	NameLengthMax = 256

	// Maximum length of a comment
	CommentLengthMax = 128
)

// Object is the interface for an nftables object. All of the concrete object types
// implement this interface.
type Object interface {
	// validate validates an object for an operation
	validate(verb verb) error

	// writeOperation writes out an "nft" operation involving the object. It assumes
	// that the object has been validated.
	writeOperation(verb verb, ctx *nftContext, writer io.Writer)
}

// Family is an nftables family
type Family string

const (
	// IPv4Family represents the "ip" nftables family, for IPv4 rules.
	IPv4Family Family = "ip"

	// IPv6Family represents the "ip6" nftables family, for IPv6 rules.
	IPv6Family Family = "ip6"

	// InetFamily represents the "inet" nftables family, for mixed IPv4 and IPv6 rules.
	InetFamily Family = "inet"

	// ARPFamily represents the "arp" nftables family, for ARP rules.
	ARPFamily Family = "arp"

	// BridgeFamily represents the "bridge" nftables family, for rules operating
	// on packets traversing a bridge.
	BridgeFamily Family = "bridge"

	// NetDevFamily represents the "netdev" nftables family, for rules operating on
	// the device ingress/egress path.
	NetDevFamily Family = "netdev"
)

// Table represents an nftables table.
type Table struct {
	// Comment is an optional comment for the table. (Note that this can be specified
	// on creation, but depending on the version of /sbin/nft that is available, it
	// may not be filled in correctly in the result of a List.)
	Comment *string

	// Handle is an identifier that can be used to uniquely identify an object when
	// deleting it. When adding a new object, this must be nil.
	Handle *int
}

// BaseChainType represents the "type" of a "base chain" (ie, a chain that is attached to a hook)
type BaseChainType string

const (
	// FilterType is the chain type for basic packet filtering.
	FilterType BaseChainType = "filter"

	// NATType is the chain type for doing DNAT, SNAT, and masquerading.
	// NAT operations are only available from certain hooks.
	NATType BaseChainType = "nat"

	// RouteType is the chain type for rules that change the routing of packets.
	// Chains of this type can only be added to the "output" hook.
	RouteType BaseChainType = "route"
)

// BaseChainHook represents the "hook" that a base chain is attached to
type BaseChainHook string

// FIXME: document these correctly; virtually all of the existing iptables/nftables
// documentation is slightly wrong, particular wrt locally-generated packets.
const (
	PreroutingHook  BaseChainHook = "prerouting"
	InputHook       BaseChainHook = "input"
	ForwardHook     BaseChainHook = "forward"
	OutputHook      BaseChainHook = "output"
	PostroutingHook BaseChainHook = "postrouting"
	IngressHook     BaseChainHook = "ingress"
	EgressHook      BaseChainHook = "egress"
)

// BaseChainPriority represents the "priority" of a base chain. In addition to the const
// values, you can also use a signed integer value, or an arithmetic expression consisting
// of a const value followed by "+" or "-" and an integer. Lower values run earlier.
type BaseChainPriority string

const (
	// RawPriority is the earliest named priority. In particular, it can be used for
	// rules that need to run before conntrack. It is equivalent to the value -300 and
	// can be used in the ip, ip6, and inet families.
	RawPriority BaseChainPriority = "raw"

	// ManglePriority is the standard priority for packet-rewriting operations. It is
	// equivalent to the value -150 and can be used in the ip, ip6, and inet families.
	ManglePriority BaseChainPriority = "mangle"

	// DNATPriority is the standard priority for DNAT operations. In the ip, ip6, and
	// inet families, it is equivalent to the value -100. In the bridge family it is
	// equivalent to the value -300. In both cases it can only be used from the
	// prerouting hook.
	DNATPriority BaseChainPriority = "dstnat"

	// FilterPriority is the standard priority for filtering operations. In the ip,
	// ip6, inet, arp, and netdev families, it is equivalent to the value 0. In the
	// bridge family it is equivalent to the value -200.
	FilterPriority BaseChainPriority = "filter"

	// OutPriority is FIXME. It is equivalent to the value 300 and can only be used in
	// the bridge family.
	OutPriority BaseChainPriority = "out"

	// SecurityPriority is the standard priority for security operations ("where
	// secmark can be set for example"). It is equivalent to the value 50 and can be
	// used in the ip, ip6, and inet families.
	SecurityPriority BaseChainPriority = "security"

	// SNATPriority is the standard priority for SNAT operations. In the ip, ip6, and
	// inet families, it is equivalent to the value 100. In the bridge family it is
	// equivalent to the value 300. In both cases it can only be used from the
	// postrouting hook.
	SNATPriority BaseChainPriority = "srcnat"
)

// Chain represents an nftables chain; either a "base chain" (if Type, Hook, and Priority
// are specified), or a "regular chain" (if they are not).
type Chain struct {
	// Name is the name of the chain.
	Name string

	// Type is the chain type; this must be set for a base chain and unset for a
	// regular chain.
	Type *BaseChainType
	// Hook is the hook that the chain is connected to; this must be set for a base
	// chain and unset for a regular chain.
	Hook *BaseChainHook
	// Priority is the chain priority; this must be set for a base chain and unset for
	// a regular chain. You can call ParsePriority() to convert this to a number.
	Priority *BaseChainPriority

	// Comment is an optional comment for the object.
	Comment *string

	// Handle is an identifier that can be used to uniquely identify an object when
	// deleting it. When adding a new object, this must be nil
	Handle *int
}

// Rule represents a rule in a chain
type Rule struct {
	// Chain is the name of the chain that contains this rule
	Chain string

	// Rule is the rule in standard nftables syntax. (Should be empty on Delete, but
	// is ignored if not.) Note that this does not include any rule comment, which is
	// separate from the rule itself.
	Rule string

	// Comment is an optional comment for the rule.
	Comment *string

	// Index is the number of a rule (counting from 0) to Add this Rule after or
	// Insert it before. Cannot be specified along with Handle. If neither Index
	// nor Handle is specified then Add appends the rule the end of the chain and
	// Insert prepends it to the beginning.
	Index *int

	// Handle is a rule handle. In Add or Insert, if set, this is the handle of
	// existing rule to put the new rule after/before. In Delete or Replace, this
	// indicates the existing rule to delete/replace, and is mandatory. In the result
	// of a List, this will indicate the rule's handle that can then be used in a
	// later operation.
	Handle *int
}

// SetFlag represents a set or map flag
type SetFlag string

const (
	// ConstantFlag is a flag indicating that the set/map is constant. FIXME UNDOCUMENTED
	ConstantFlag SetFlag = "constant"

	// DynamicFlag is a flag indicating that the set contains stateful objects
	// (counters, quotas, or limits) that will be dynamically updated.
	DynamicFlag SetFlag = "dynamic"

	// IntervalFlag is a flag indicating that the set contains either CIDR elements or
	// IP ranges.
	IntervalFlag SetFlag = "interval"

	// TimeoutFlag is a flag indicating that the set/map has a timeout after which
	// dynamically added elements will be removed. (It is set automatically if the
	// set/map has a Timeout.)
	TimeoutFlag SetFlag = "timeout"
)

// SetPolicy represents a set or map storage policy
type SetPolicy string

const (
	// PolicyPerformance FIXME
	PerformancePolicy SetPolicy = "performance"

	// PolicyMemory FIXME
	MemoryPolicy SetPolicy = "memory"
)

// Set represents the definition of an nftables set (but not its elements)
type Set struct {
	// Name is the name of the set.
	Name string

	// Type is the type of the set key (eg "ipv4_addr"). Either Type or TypeOf, but
	// not both, must be non-empty.
	Type string

	// TypeOf is the type of the set key as an nftables expression (eg "ip saddr").
	// Either Type or TypeOf, but not both, must be non-empty.
	TypeOf string

	// Flags are the set flags
	Flags []SetFlag

	// Timeout is the time that an element will stay in the set before being removed.
	// (Optional; mandatory for sets that will be added to from the packet path)
	Timeout *time.Duration

	// GCInterval is the interval at which timed-out elements will be removed from the
	// set. (Optional; FIXME DEFAULT)
	GCInterval *time.Duration

	// Size if the maximum numer of elements in the set.
	// (Optional; mandatory for sets that will be added to from the packet path)
	Size *uint64

	// Policy is the FIXME
	Policy *SetPolicy

	// AutoMerge indicates that adjacent/overlapping set elements should be merged
	// together (only for interval sets)
	AutoMerge *bool

	// Comment is an optional comment for the object.
	Comment *string

	// Handle is an identifier that can be used to uniquely identify an object when
	// deleting it. When adding a new object, this must be nil
	Handle *int
}

// Map represents the definition of an nftables map (but not its elements)
type Map struct {
	// Name is the name of the map.
	Name string

	// Type is the type of the map key and value (eg "ipv4_addr : verdict"). Either
	// Type or TypeOf, but not both, must be non-empty.
	Type string

	// TypeOf is the type of the set key as an nftables expression (eg "ip saddr : verdict").
	// Either Type or TypeOf, but not both, must be non-empty.
	TypeOf string

	// Flags are the map flags
	Flags []SetFlag

	// Timeout is the time that an element will stay in the set before being removed.
	// (Optional; mandatory for sets that will be added to from the packet path)
	Timeout *time.Duration

	// GCInterval is the interval at which timed-out elements will be removed from the
	// set. (Optional; FIXME DEFAULT)
	GCInterval *time.Duration

	// Size if the maximum numer of elements in the set.
	// (Optional; mandatory for sets that will be added to from the packet path)
	Size *uint64

	// Policy is the FIXME
	Policy *SetPolicy

	// Comment is an optional comment for the object.
	Comment *string

	// Handle is an identifier that can be used to uniquely identify an object when
	// deleting it. When adding a new object, this must be nil
	Handle *int
}

// Element represents a set or map element
type Element struct {
	// Set is the name of the set that contains this element (or the empty string if
	// this is a map element.)
	Set string

	// Map is the name of the map that contains this element (or the empty string if
	// this is a set element.)
	Map string

	// Key is the element key. (The list contains a single element for "simple" keys,
	// or multiple elements for concatenations.)
	Key []string

	// Value is the map element value. As with Key, this may be a single value or
	// multiple. For set elements, this must be nil.
	Value []string

	// Comment is an optional comment for the element
	Comment *string
}
