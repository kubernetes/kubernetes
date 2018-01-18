package nl

import (
	"fmt"
	"unsafe"
)

const (
	SizeofBridgeVlanInfo = 0x04
)

/* Bridge Flags */
const (
	BRIDGE_FLAGS_MASTER = iota /* Bridge command to/from master */
	BRIDGE_FLAGS_SELF          /* Bridge command to/from lowerdev */
)

/* Bridge management nested attributes
 * [IFLA_AF_SPEC] = {
 *     [IFLA_BRIDGE_FLAGS]
 *     [IFLA_BRIDGE_MODE]
 *     [IFLA_BRIDGE_VLAN_INFO]
 * }
 */
const (
	IFLA_BRIDGE_FLAGS = iota
	IFLA_BRIDGE_MODE
	IFLA_BRIDGE_VLAN_INFO
)

const (
	BRIDGE_VLAN_INFO_MASTER = 1 << iota
	BRIDGE_VLAN_INFO_PVID
	BRIDGE_VLAN_INFO_UNTAGGED
	BRIDGE_VLAN_INFO_RANGE_BEGIN
	BRIDGE_VLAN_INFO_RANGE_END
)

// struct bridge_vlan_info {
//   __u16 flags;
//   __u16 vid;
// };

type BridgeVlanInfo struct {
	Flags uint16
	Vid   uint16
}

func (b *BridgeVlanInfo) Serialize() []byte {
	return (*(*[SizeofBridgeVlanInfo]byte)(unsafe.Pointer(b)))[:]
}

func DeserializeBridgeVlanInfo(b []byte) *BridgeVlanInfo {
	return (*BridgeVlanInfo)(unsafe.Pointer(&b[0:SizeofBridgeVlanInfo][0]))
}

func (b *BridgeVlanInfo) PortVID() bool {
	return b.Flags&BRIDGE_VLAN_INFO_PVID > 0
}

func (b *BridgeVlanInfo) EngressUntag() bool {
	return b.Flags&BRIDGE_VLAN_INFO_UNTAGGED > 0
}

func (b *BridgeVlanInfo) String() string {
	return fmt.Sprintf("%+v", *b)
}

/* New extended info filters for IFLA_EXT_MASK */
const (
	RTEXT_FILTER_VF = 1 << iota
	RTEXT_FILTER_BRVLAN
	RTEXT_FILTER_BRVLAN_COMPRESSED
)
