package netlink

import (
	"fmt"
	"net"
	"os"
)

// Link represents a link device from netlink. Shared link attributes
// like name may be retrieved using the Attrs() method. Unique data
// can be retrieved by casting the object to the proper type.
type Link interface {
	Attrs() *LinkAttrs
	Type() string
}

type (
	NsPid int
	NsFd  int
)

// LinkAttrs represents data shared by most link types
type LinkAttrs struct {
	Index        int
	MTU          int
	TxQLen       int // Transmit Queue Length
	Name         string
	HardwareAddr net.HardwareAddr
	Flags        net.Flags
	RawFlags     uint32
	ParentIndex  int         // index of the parent link device
	MasterIndex  int         // must be the index of a bridge
	Namespace    interface{} // nil | NsPid | NsFd
	Alias        string
	Statistics   *LinkStatistics
	Promisc      int
	Xdp          *LinkXdp
	EncapType    string
	Protinfo     *Protinfo
	OperState    LinkOperState
	NetNsID      int
	NumTxQueues  int
	NumRxQueues  int
}

// LinkOperState represents the values of the IFLA_OPERSTATE link
// attribute, which contains the RFC2863 state of the interface.
type LinkOperState uint8

const (
	OperUnknown        = iota // Status can't be determined.
	OperNotPresent            // Some component is missing.
	OperDown                  // Down.
	OperLowerLayerDown        // Down due to state of lower layer.
	OperTesting               // In some test mode.
	OperDormant               // Not up but pending an external event.
	OperUp                    // Up, ready to send packets.
)

func (s LinkOperState) String() string {
	switch s {
	case OperNotPresent:
		return "not-present"
	case OperDown:
		return "down"
	case OperLowerLayerDown:
		return "lower-layer-down"
	case OperTesting:
		return "testing"
	case OperDormant:
		return "dormant"
	case OperUp:
		return "up"
	default:
		return "unknown"
	}
}

// NewLinkAttrs returns LinkAttrs structure filled with default values
func NewLinkAttrs() LinkAttrs {
	return LinkAttrs{
		TxQLen: -1,
	}
}

type LinkStatistics LinkStatistics64

/*
Ref: struct rtnl_link_stats {...}
*/
type LinkStatistics32 struct {
	RxPackets         uint32
	TxPackets         uint32
	RxBytes           uint32
	TxBytes           uint32
	RxErrors          uint32
	TxErrors          uint32
	RxDropped         uint32
	TxDropped         uint32
	Multicast         uint32
	Collisions        uint32
	RxLengthErrors    uint32
	RxOverErrors      uint32
	RxCrcErrors       uint32
	RxFrameErrors     uint32
	RxFifoErrors      uint32
	RxMissedErrors    uint32
	TxAbortedErrors   uint32
	TxCarrierErrors   uint32
	TxFifoErrors      uint32
	TxHeartbeatErrors uint32
	TxWindowErrors    uint32
	RxCompressed      uint32
	TxCompressed      uint32
}

func (s32 LinkStatistics32) to64() *LinkStatistics64 {
	return &LinkStatistics64{
		RxPackets:         uint64(s32.RxPackets),
		TxPackets:         uint64(s32.TxPackets),
		RxBytes:           uint64(s32.RxBytes),
		TxBytes:           uint64(s32.TxBytes),
		RxErrors:          uint64(s32.RxErrors),
		TxErrors:          uint64(s32.TxErrors),
		RxDropped:         uint64(s32.RxDropped),
		TxDropped:         uint64(s32.TxDropped),
		Multicast:         uint64(s32.Multicast),
		Collisions:        uint64(s32.Collisions),
		RxLengthErrors:    uint64(s32.RxLengthErrors),
		RxOverErrors:      uint64(s32.RxOverErrors),
		RxCrcErrors:       uint64(s32.RxCrcErrors),
		RxFrameErrors:     uint64(s32.RxFrameErrors),
		RxFifoErrors:      uint64(s32.RxFifoErrors),
		RxMissedErrors:    uint64(s32.RxMissedErrors),
		TxAbortedErrors:   uint64(s32.TxAbortedErrors),
		TxCarrierErrors:   uint64(s32.TxCarrierErrors),
		TxFifoErrors:      uint64(s32.TxFifoErrors),
		TxHeartbeatErrors: uint64(s32.TxHeartbeatErrors),
		TxWindowErrors:    uint64(s32.TxWindowErrors),
		RxCompressed:      uint64(s32.RxCompressed),
		TxCompressed:      uint64(s32.TxCompressed),
	}
}

/*
Ref: struct rtnl_link_stats64 {...}
*/
type LinkStatistics64 struct {
	RxPackets         uint64
	TxPackets         uint64
	RxBytes           uint64
	TxBytes           uint64
	RxErrors          uint64
	TxErrors          uint64
	RxDropped         uint64
	TxDropped         uint64
	Multicast         uint64
	Collisions        uint64
	RxLengthErrors    uint64
	RxOverErrors      uint64
	RxCrcErrors       uint64
	RxFrameErrors     uint64
	RxFifoErrors      uint64
	RxMissedErrors    uint64
	TxAbortedErrors   uint64
	TxCarrierErrors   uint64
	TxFifoErrors      uint64
	TxHeartbeatErrors uint64
	TxWindowErrors    uint64
	RxCompressed      uint64
	TxCompressed      uint64
}

type LinkXdp struct {
	Fd       int
	Attached bool
	Flags    uint32
	ProgId   uint32
}

// Device links cannot be created via netlink. These links
// are links created by udev like 'lo' and 'etho0'
type Device struct {
	LinkAttrs
}

func (device *Device) Attrs() *LinkAttrs {
	return &device.LinkAttrs
}

func (device *Device) Type() string {
	return "device"
}

// Dummy links are dummy ethernet devices
type Dummy struct {
	LinkAttrs
}

func (dummy *Dummy) Attrs() *LinkAttrs {
	return &dummy.LinkAttrs
}

func (dummy *Dummy) Type() string {
	return "dummy"
}

// Ifb links are advanced dummy devices for packet filtering
type Ifb struct {
	LinkAttrs
}

func (ifb *Ifb) Attrs() *LinkAttrs {
	return &ifb.LinkAttrs
}

func (ifb *Ifb) Type() string {
	return "ifb"
}

// Bridge links are simple linux bridges
type Bridge struct {
	LinkAttrs
	MulticastSnooping *bool
	HelloTime         *uint32
}

func (bridge *Bridge) Attrs() *LinkAttrs {
	return &bridge.LinkAttrs
}

func (bridge *Bridge) Type() string {
	return "bridge"
}

// Vlan links have ParentIndex set in their Attrs()
type Vlan struct {
	LinkAttrs
	VlanId int
}

func (vlan *Vlan) Attrs() *LinkAttrs {
	return &vlan.LinkAttrs
}

func (vlan *Vlan) Type() string {
	return "vlan"
}

type MacvlanMode uint16

const (
	MACVLAN_MODE_DEFAULT MacvlanMode = iota
	MACVLAN_MODE_PRIVATE
	MACVLAN_MODE_VEPA
	MACVLAN_MODE_BRIDGE
	MACVLAN_MODE_PASSTHRU
	MACVLAN_MODE_SOURCE
)

// Macvlan links have ParentIndex set in their Attrs()
type Macvlan struct {
	LinkAttrs
	Mode MacvlanMode

	// MACAddrs is only populated for Macvlan SOURCE links
	MACAddrs []net.HardwareAddr
}

func (macvlan *Macvlan) Attrs() *LinkAttrs {
	return &macvlan.LinkAttrs
}

func (macvlan *Macvlan) Type() string {
	return "macvlan"
}

// Macvtap - macvtap is a virtual interfaces based on macvlan
type Macvtap struct {
	Macvlan
}

func (macvtap Macvtap) Type() string {
	return "macvtap"
}

type TuntapMode uint16
type TuntapFlag uint16

// Tuntap links created via /dev/tun/tap, but can be destroyed via netlink
type Tuntap struct {
	LinkAttrs
	Mode   TuntapMode
	Flags  TuntapFlag
	Queues int
	Fds    []*os.File
}

func (tuntap *Tuntap) Attrs() *LinkAttrs {
	return &tuntap.LinkAttrs
}

func (tuntap *Tuntap) Type() string {
	return "tuntap"
}

// Veth devices must specify PeerName on create
type Veth struct {
	LinkAttrs
	PeerName string // veth on create only
}

func (veth *Veth) Attrs() *LinkAttrs {
	return &veth.LinkAttrs
}

func (veth *Veth) Type() string {
	return "veth"
}

// GenericLink links represent types that are not currently understood
// by this netlink library.
type GenericLink struct {
	LinkAttrs
	LinkType string
}

func (generic *GenericLink) Attrs() *LinkAttrs {
	return &generic.LinkAttrs
}

func (generic *GenericLink) Type() string {
	return generic.LinkType
}

type Vxlan struct {
	LinkAttrs
	VxlanId        int
	VtepDevIndex   int
	SrcAddr        net.IP
	Group          net.IP
	TTL            int
	TOS            int
	Learning       bool
	Proxy          bool
	RSC            bool
	L2miss         bool
	L3miss         bool
	UDPCSum        bool
	UDP6ZeroCSumTx bool
	UDP6ZeroCSumRx bool
	NoAge          bool
	GBP            bool
	FlowBased      bool
	Age            int
	Limit          int
	Port           int
	PortLow        int
	PortHigh       int
}

func (vxlan *Vxlan) Attrs() *LinkAttrs {
	return &vxlan.LinkAttrs
}

func (vxlan *Vxlan) Type() string {
	return "vxlan"
}

type IPVlanMode uint16

const (
	IPVLAN_MODE_L2 IPVlanMode = iota
	IPVLAN_MODE_L3
	IPVLAN_MODE_L3S
	IPVLAN_MODE_MAX
)

type IPVlan struct {
	LinkAttrs
	Mode IPVlanMode
}

func (ipvlan *IPVlan) Attrs() *LinkAttrs {
	return &ipvlan.LinkAttrs
}

func (ipvlan *IPVlan) Type() string {
	return "ipvlan"
}

// BondMode type
type BondMode int

func (b BondMode) String() string {
	s, ok := bondModeToString[b]
	if !ok {
		return fmt.Sprintf("BondMode(%d)", b)
	}
	return s
}

// StringToBondMode returns bond mode, or uknonw is the s is invalid.
func StringToBondMode(s string) BondMode {
	mode, ok := StringToBondModeMap[s]
	if !ok {
		return BOND_MODE_UNKNOWN
	}
	return mode
}

// Possible BondMode
const (
	BOND_MODE_BALANCE_RR BondMode = iota
	BOND_MODE_ACTIVE_BACKUP
	BOND_MODE_BALANCE_XOR
	BOND_MODE_BROADCAST
	BOND_MODE_802_3AD
	BOND_MODE_BALANCE_TLB
	BOND_MODE_BALANCE_ALB
	BOND_MODE_UNKNOWN
)

var bondModeToString = map[BondMode]string{
	BOND_MODE_BALANCE_RR:    "balance-rr",
	BOND_MODE_ACTIVE_BACKUP: "active-backup",
	BOND_MODE_BALANCE_XOR:   "balance-xor",
	BOND_MODE_BROADCAST:     "broadcast",
	BOND_MODE_802_3AD:       "802.3ad",
	BOND_MODE_BALANCE_TLB:   "balance-tlb",
	BOND_MODE_BALANCE_ALB:   "balance-alb",
}
var StringToBondModeMap = map[string]BondMode{
	"balance-rr":    BOND_MODE_BALANCE_RR,
	"active-backup": BOND_MODE_ACTIVE_BACKUP,
	"balance-xor":   BOND_MODE_BALANCE_XOR,
	"broadcast":     BOND_MODE_BROADCAST,
	"802.3ad":       BOND_MODE_802_3AD,
	"balance-tlb":   BOND_MODE_BALANCE_TLB,
	"balance-alb":   BOND_MODE_BALANCE_ALB,
}

// BondArpValidate type
type BondArpValidate int

// Possible BondArpValidate value
const (
	BOND_ARP_VALIDATE_NONE BondArpValidate = iota
	BOND_ARP_VALIDATE_ACTIVE
	BOND_ARP_VALIDATE_BACKUP
	BOND_ARP_VALIDATE_ALL
)

// BondPrimaryReselect type
type BondPrimaryReselect int

// Possible BondPrimaryReselect value
const (
	BOND_PRIMARY_RESELECT_ALWAYS BondPrimaryReselect = iota
	BOND_PRIMARY_RESELECT_BETTER
	BOND_PRIMARY_RESELECT_FAILURE
)

// BondArpAllTargets type
type BondArpAllTargets int

// Possible BondArpAllTargets value
const (
	BOND_ARP_ALL_TARGETS_ANY BondArpAllTargets = iota
	BOND_ARP_ALL_TARGETS_ALL
)

// BondFailOverMac type
type BondFailOverMac int

// Possible BondFailOverMac value
const (
	BOND_FAIL_OVER_MAC_NONE BondFailOverMac = iota
	BOND_FAIL_OVER_MAC_ACTIVE
	BOND_FAIL_OVER_MAC_FOLLOW
)

// BondXmitHashPolicy type
type BondXmitHashPolicy int

func (b BondXmitHashPolicy) String() string {
	s, ok := bondXmitHashPolicyToString[b]
	if !ok {
		return fmt.Sprintf("XmitHashPolicy(%d)", b)
	}
	return s
}

// StringToBondXmitHashPolicy returns bond lacp arte, or uknonw is the s is invalid.
func StringToBondXmitHashPolicy(s string) BondXmitHashPolicy {
	lacp, ok := StringToBondXmitHashPolicyMap[s]
	if !ok {
		return BOND_XMIT_HASH_POLICY_UNKNOWN
	}
	return lacp
}

// Possible BondXmitHashPolicy value
const (
	BOND_XMIT_HASH_POLICY_LAYER2 BondXmitHashPolicy = iota
	BOND_XMIT_HASH_POLICY_LAYER3_4
	BOND_XMIT_HASH_POLICY_LAYER2_3
	BOND_XMIT_HASH_POLICY_ENCAP2_3
	BOND_XMIT_HASH_POLICY_ENCAP3_4
	BOND_XMIT_HASH_POLICY_UNKNOWN
)

var bondXmitHashPolicyToString = map[BondXmitHashPolicy]string{
	BOND_XMIT_HASH_POLICY_LAYER2:   "layer2",
	BOND_XMIT_HASH_POLICY_LAYER3_4: "layer3+4",
	BOND_XMIT_HASH_POLICY_LAYER2_3: "layer2+3",
	BOND_XMIT_HASH_POLICY_ENCAP2_3: "encap2+3",
	BOND_XMIT_HASH_POLICY_ENCAP3_4: "encap3+4",
}
var StringToBondXmitHashPolicyMap = map[string]BondXmitHashPolicy{
	"layer2":   BOND_XMIT_HASH_POLICY_LAYER2,
	"layer3+4": BOND_XMIT_HASH_POLICY_LAYER3_4,
	"layer2+3": BOND_XMIT_HASH_POLICY_LAYER2_3,
	"encap2+3": BOND_XMIT_HASH_POLICY_ENCAP2_3,
	"encap3+4": BOND_XMIT_HASH_POLICY_ENCAP3_4,
}

// BondLacpRate type
type BondLacpRate int

func (b BondLacpRate) String() string {
	s, ok := bondLacpRateToString[b]
	if !ok {
		return fmt.Sprintf("LacpRate(%d)", b)
	}
	return s
}

// StringToBondLacpRate returns bond lacp arte, or uknonw is the s is invalid.
func StringToBondLacpRate(s string) BondLacpRate {
	lacp, ok := StringToBondLacpRateMap[s]
	if !ok {
		return BOND_LACP_RATE_UNKNOWN
	}
	return lacp
}

// Possible BondLacpRate value
const (
	BOND_LACP_RATE_SLOW BondLacpRate = iota
	BOND_LACP_RATE_FAST
	BOND_LACP_RATE_UNKNOWN
)

var bondLacpRateToString = map[BondLacpRate]string{
	BOND_LACP_RATE_SLOW: "slow",
	BOND_LACP_RATE_FAST: "fast",
}
var StringToBondLacpRateMap = map[string]BondLacpRate{
	"slow": BOND_LACP_RATE_SLOW,
	"fast": BOND_LACP_RATE_FAST,
}

// BondAdSelect type
type BondAdSelect int

// Possible BondAdSelect value
const (
	BOND_AD_SELECT_STABLE BondAdSelect = iota
	BOND_AD_SELECT_BANDWIDTH
	BOND_AD_SELECT_COUNT
)

// BondAdInfo represents ad info for bond
type BondAdInfo struct {
	AggregatorId int
	NumPorts     int
	ActorKey     int
	PartnerKey   int
	PartnerMac   net.HardwareAddr
}

// Bond representation
type Bond struct {
	LinkAttrs
	Mode            BondMode
	ActiveSlave     int
	Miimon          int
	UpDelay         int
	DownDelay       int
	UseCarrier      int
	ArpInterval     int
	ArpIpTargets    []net.IP
	ArpValidate     BondArpValidate
	ArpAllTargets   BondArpAllTargets
	Primary         int
	PrimaryReselect BondPrimaryReselect
	FailOverMac     BondFailOverMac
	XmitHashPolicy  BondXmitHashPolicy
	ResendIgmp      int
	NumPeerNotif    int
	AllSlavesActive int
	MinLinks        int
	LpInterval      int
	PackersPerSlave int
	LacpRate        BondLacpRate
	AdSelect        BondAdSelect
	// looking at iproute tool AdInfo can only be retrived. It can't be set.
	AdInfo         *BondAdInfo
	AdActorSysPrio int
	AdUserPortKey  int
	AdActorSystem  net.HardwareAddr
	TlbDynamicLb   int
}

func NewLinkBond(atr LinkAttrs) *Bond {
	return &Bond{
		LinkAttrs:       atr,
		Mode:            -1,
		ActiveSlave:     -1,
		Miimon:          -1,
		UpDelay:         -1,
		DownDelay:       -1,
		UseCarrier:      -1,
		ArpInterval:     -1,
		ArpIpTargets:    nil,
		ArpValidate:     -1,
		ArpAllTargets:   -1,
		Primary:         -1,
		PrimaryReselect: -1,
		FailOverMac:     -1,
		XmitHashPolicy:  -1,
		ResendIgmp:      -1,
		NumPeerNotif:    -1,
		AllSlavesActive: -1,
		MinLinks:        -1,
		LpInterval:      -1,
		PackersPerSlave: -1,
		LacpRate:        -1,
		AdSelect:        -1,
		AdActorSysPrio:  -1,
		AdUserPortKey:   -1,
		AdActorSystem:   nil,
		TlbDynamicLb:    -1,
	}
}

// Flag mask for bond options. Bond.Flagmask must be set to on for option to work.
const (
	BOND_MODE_MASK uint64 = 1 << (1 + iota)
	BOND_ACTIVE_SLAVE_MASK
	BOND_MIIMON_MASK
	BOND_UPDELAY_MASK
	BOND_DOWNDELAY_MASK
	BOND_USE_CARRIER_MASK
	BOND_ARP_INTERVAL_MASK
	BOND_ARP_VALIDATE_MASK
	BOND_ARP_ALL_TARGETS_MASK
	BOND_PRIMARY_MASK
	BOND_PRIMARY_RESELECT_MASK
	BOND_FAIL_OVER_MAC_MASK
	BOND_XMIT_HASH_POLICY_MASK
	BOND_RESEND_IGMP_MASK
	BOND_NUM_PEER_NOTIF_MASK
	BOND_ALL_SLAVES_ACTIVE_MASK
	BOND_MIN_LINKS_MASK
	BOND_LP_INTERVAL_MASK
	BOND_PACKETS_PER_SLAVE_MASK
	BOND_LACP_RATE_MASK
	BOND_AD_SELECT_MASK
)

// Attrs implementation.
func (bond *Bond) Attrs() *LinkAttrs {
	return &bond.LinkAttrs
}

// Type implementation fro Vxlan.
func (bond *Bond) Type() string {
	return "bond"
}

// Gretap devices must specify LocalIP and RemoteIP on create
type Gretap struct {
	LinkAttrs
	IKey       uint32
	OKey       uint32
	EncapSport uint16
	EncapDport uint16
	Local      net.IP
	Remote     net.IP
	IFlags     uint16
	OFlags     uint16
	PMtuDisc   uint8
	Ttl        uint8
	Tos        uint8
	EncapType  uint16
	EncapFlags uint16
	Link       uint32
	FlowBased  bool
}

func (gretap *Gretap) Attrs() *LinkAttrs {
	return &gretap.LinkAttrs
}

func (gretap *Gretap) Type() string {
	return "gretap"
}

type Iptun struct {
	LinkAttrs
	Ttl        uint8
	Tos        uint8
	PMtuDisc   uint8
	Link       uint32
	Local      net.IP
	Remote     net.IP
	EncapSport uint16
	EncapDport uint16
	EncapType  uint16
	EncapFlags uint16
	FlowBased  bool
}

func (iptun *Iptun) Attrs() *LinkAttrs {
	return &iptun.LinkAttrs
}

func (iptun *Iptun) Type() string {
	return "ipip"
}

type Sittun struct {
	LinkAttrs
	Link       uint32
	Local      net.IP
	Remote     net.IP
	Ttl        uint8
	Tos        uint8
	PMtuDisc   uint8
	EncapType  uint16
	EncapFlags uint16
	EncapSport uint16
	EncapDport uint16
}

func (sittun *Sittun) Attrs() *LinkAttrs {
	return &sittun.LinkAttrs
}

func (sittun *Sittun) Type() string {
	return "sit"
}

type Vti struct {
	LinkAttrs
	IKey   uint32
	OKey   uint32
	Link   uint32
	Local  net.IP
	Remote net.IP
}

func (vti *Vti) Attrs() *LinkAttrs {
	return &vti.LinkAttrs
}

func (iptun *Vti) Type() string {
	return "vti"
}

type Gretun struct {
	LinkAttrs
	Link       uint32
	IFlags     uint16
	OFlags     uint16
	IKey       uint32
	OKey       uint32
	Local      net.IP
	Remote     net.IP
	Ttl        uint8
	Tos        uint8
	PMtuDisc   uint8
	EncapType  uint16
	EncapFlags uint16
	EncapSport uint16
	EncapDport uint16
}

func (gretun *Gretun) Attrs() *LinkAttrs {
	return &gretun.LinkAttrs
}

func (gretun *Gretun) Type() string {
	return "gre"
}

type Vrf struct {
	LinkAttrs
	Table uint32
}

func (vrf *Vrf) Attrs() *LinkAttrs {
	return &vrf.LinkAttrs
}

func (vrf *Vrf) Type() string {
	return "vrf"
}

type GTP struct {
	LinkAttrs
	FD0         int
	FD1         int
	Role        int
	PDPHashsize int
}

func (gtp *GTP) Attrs() *LinkAttrs {
	return &gtp.LinkAttrs
}

func (gtp *GTP) Type() string {
	return "gtp"
}

// iproute2 supported devices;
// vlan | veth | vcan | dummy | ifb | macvlan | macvtap |
// bridge | bond | ipoib | ip6tnl | ipip | sit | vxlan |
// gre | gretap | ip6gre | ip6gretap | vti | nlmon |
// bond_slave | ipvlan

// LinkNotFoundError wraps the various not found errors when
// getting/reading links. This is intended for better error
// handling by dependent code so that "not found error" can
// be distinguished from other errors
type LinkNotFoundError struct {
	error
}
