package netlink

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net"
	"os"
	"syscall"
	"unsafe"

	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
)

const (
	SizeofLinkStats32 = 0x5c
	SizeofLinkStats64 = 0xd8
	IFLA_STATS64      = 0x17 // syscall pkg does not contain this one
)

const (
	TUNTAP_MODE_TUN  TuntapMode = syscall.IFF_TUN
	TUNTAP_MODE_TAP  TuntapMode = syscall.IFF_TAP
	TUNTAP_DEFAULTS  TuntapFlag = syscall.IFF_TUN_EXCL | syscall.IFF_ONE_QUEUE
	TUNTAP_VNET_HDR  TuntapFlag = syscall.IFF_VNET_HDR
	TUNTAP_TUN_EXCL  TuntapFlag = syscall.IFF_TUN_EXCL
	TUNTAP_NO_PI     TuntapFlag = syscall.IFF_NO_PI
	TUNTAP_ONE_QUEUE TuntapFlag = syscall.IFF_ONE_QUEUE
)

var lookupByDump = false

var macvlanModes = [...]uint32{
	0,
	nl.MACVLAN_MODE_PRIVATE,
	nl.MACVLAN_MODE_VEPA,
	nl.MACVLAN_MODE_BRIDGE,
	nl.MACVLAN_MODE_PASSTHRU,
	nl.MACVLAN_MODE_SOURCE,
}

func ensureIndex(link *LinkAttrs) {
	if link != nil && link.Index == 0 {
		newlink, _ := LinkByName(link.Name)
		if newlink != nil {
			link.Index = newlink.Attrs().Index
		}
	}
}

func (h *Handle) ensureIndex(link *LinkAttrs) {
	if link != nil && link.Index == 0 {
		newlink, _ := h.LinkByName(link.Name)
		if newlink != nil {
			link.Index = newlink.Attrs().Index
		}
	}
}

func (h *Handle) LinkSetARPOff(link Link) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Change |= syscall.IFF_NOARP
	msg.Flags |= syscall.IFF_NOARP
	msg.Index = int32(base.Index)
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

func LinkSetARPOff(link Link) error {
	return pkgHandle.LinkSetARPOff(link)
}

func (h *Handle) LinkSetARPOn(link Link) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Change |= syscall.IFF_NOARP
	msg.Flags &= ^uint32(syscall.IFF_NOARP)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

func LinkSetARPOn(link Link) error {
	return pkgHandle.LinkSetARPOn(link)
}

func (h *Handle) SetPromiscOn(link Link) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Change = syscall.IFF_PROMISC
	msg.Flags = syscall.IFF_PROMISC
	msg.Index = int32(base.Index)
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

func BridgeSetMcastSnoop(link Link, on bool) error {
	return pkgHandle.BridgeSetMcastSnoop(link, on)
}

func (h *Handle) BridgeSetMcastSnoop(link Link, on bool) error {
	bridge := link.(*Bridge)
	bridge.MulticastSnooping = &on
	return h.linkModify(bridge, syscall.NLM_F_ACK)
}

func SetPromiscOn(link Link) error {
	return pkgHandle.SetPromiscOn(link)
}

func (h *Handle) SetPromiscOff(link Link) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Change = syscall.IFF_PROMISC
	msg.Flags = 0 & ^syscall.IFF_PROMISC
	msg.Index = int32(base.Index)
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

func SetPromiscOff(link Link) error {
	return pkgHandle.SetPromiscOff(link)
}

// LinkSetUp enables the link device.
// Equivalent to: `ip link set $link up`
func LinkSetUp(link Link) error {
	return pkgHandle.LinkSetUp(link)
}

// LinkSetUp enables the link device.
// Equivalent to: `ip link set $link up`
func (h *Handle) LinkSetUp(link Link) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_NEWLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Change = syscall.IFF_UP
	msg.Flags = syscall.IFF_UP
	msg.Index = int32(base.Index)
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetDown disables link device.
// Equivalent to: `ip link set $link down`
func LinkSetDown(link Link) error {
	return pkgHandle.LinkSetDown(link)
}

// LinkSetDown disables link device.
// Equivalent to: `ip link set $link down`
func (h *Handle) LinkSetDown(link Link) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_NEWLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Change = syscall.IFF_UP
	msg.Flags = 0 & ^syscall.IFF_UP
	msg.Index = int32(base.Index)
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetMTU sets the mtu of the link device.
// Equivalent to: `ip link set $link mtu $mtu`
func LinkSetMTU(link Link, mtu int) error {
	return pkgHandle.LinkSetMTU(link, mtu)
}

// LinkSetMTU sets the mtu of the link device.
// Equivalent to: `ip link set $link mtu $mtu`
func (h *Handle) LinkSetMTU(link Link, mtu int) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	b := make([]byte, 4)
	native.PutUint32(b, uint32(mtu))

	data := nl.NewRtAttr(syscall.IFLA_MTU, b)
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetName sets the name of the link device.
// Equivalent to: `ip link set $link name $name`
func LinkSetName(link Link, name string) error {
	return pkgHandle.LinkSetName(link, name)
}

// LinkSetName sets the name of the link device.
// Equivalent to: `ip link set $link name $name`
func (h *Handle) LinkSetName(link Link, name string) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	data := nl.NewRtAttr(syscall.IFLA_IFNAME, []byte(name))
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetAlias sets the alias of the link device.
// Equivalent to: `ip link set dev $link alias $name`
func LinkSetAlias(link Link, name string) error {
	return pkgHandle.LinkSetAlias(link, name)
}

// LinkSetAlias sets the alias of the link device.
// Equivalent to: `ip link set dev $link alias $name`
func (h *Handle) LinkSetAlias(link Link, name string) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	data := nl.NewRtAttr(syscall.IFLA_IFALIAS, []byte(name))
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetHardwareAddr sets the hardware address of the link device.
// Equivalent to: `ip link set $link address $hwaddr`
func LinkSetHardwareAddr(link Link, hwaddr net.HardwareAddr) error {
	return pkgHandle.LinkSetHardwareAddr(link, hwaddr)
}

// LinkSetHardwareAddr sets the hardware address of the link device.
// Equivalent to: `ip link set $link address $hwaddr`
func (h *Handle) LinkSetHardwareAddr(link Link, hwaddr net.HardwareAddr) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	data := nl.NewRtAttr(syscall.IFLA_ADDRESS, []byte(hwaddr))
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetVfHardwareAddr sets the hardware address of a vf for the link.
// Equivalent to: `ip link set $link vf $vf mac $hwaddr`
func LinkSetVfHardwareAddr(link Link, vf int, hwaddr net.HardwareAddr) error {
	return pkgHandle.LinkSetVfHardwareAddr(link, vf, hwaddr)
}

// LinkSetVfHardwareAddr sets the hardware address of a vf for the link.
// Equivalent to: `ip link set $link vf $vf mac $hwaddr`
func (h *Handle) LinkSetVfHardwareAddr(link Link, vf int, hwaddr net.HardwareAddr) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	data := nl.NewRtAttr(nl.IFLA_VFINFO_LIST, nil)
	info := nl.NewRtAttrChild(data, nl.IFLA_VF_INFO, nil)
	vfmsg := nl.VfMac{
		Vf: uint32(vf),
	}
	copy(vfmsg.Mac[:], []byte(hwaddr))
	nl.NewRtAttrChild(info, nl.IFLA_VF_MAC, vfmsg.Serialize())
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetVfVlan sets the vlan of a vf for the link.
// Equivalent to: `ip link set $link vf $vf vlan $vlan`
func LinkSetVfVlan(link Link, vf, vlan int) error {
	return pkgHandle.LinkSetVfVlan(link, vf, vlan)
}

// LinkSetVfVlan sets the vlan of a vf for the link.
// Equivalent to: `ip link set $link vf $vf vlan $vlan`
func (h *Handle) LinkSetVfVlan(link Link, vf, vlan int) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	data := nl.NewRtAttr(nl.IFLA_VFINFO_LIST, nil)
	info := nl.NewRtAttrChild(data, nl.IFLA_VF_INFO, nil)
	vfmsg := nl.VfVlan{
		Vf:   uint32(vf),
		Vlan: uint32(vlan),
	}
	nl.NewRtAttrChild(info, nl.IFLA_VF_VLAN, vfmsg.Serialize())
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetVfTxRate sets the tx rate of a vf for the link.
// Equivalent to: `ip link set $link vf $vf rate $rate`
func LinkSetVfTxRate(link Link, vf, rate int) error {
	return pkgHandle.LinkSetVfTxRate(link, vf, rate)
}

// LinkSetVfTxRate sets the tx rate of a vf for the link.
// Equivalent to: `ip link set $link vf $vf rate $rate`
func (h *Handle) LinkSetVfTxRate(link Link, vf, rate int) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	data := nl.NewRtAttr(nl.IFLA_VFINFO_LIST, nil)
	info := nl.NewRtAttrChild(data, nl.IFLA_VF_INFO, nil)
	vfmsg := nl.VfTxRate{
		Vf:   uint32(vf),
		Rate: uint32(rate),
	}
	nl.NewRtAttrChild(info, nl.IFLA_VF_TX_RATE, vfmsg.Serialize())
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetMaster sets the master of the link device.
// Equivalent to: `ip link set $link master $master`
func LinkSetMaster(link Link, master *Bridge) error {
	return pkgHandle.LinkSetMaster(link, master)
}

// LinkSetMaster sets the master of the link device.
// Equivalent to: `ip link set $link master $master`
func (h *Handle) LinkSetMaster(link Link, master *Bridge) error {
	index := 0
	if master != nil {
		masterBase := master.Attrs()
		h.ensureIndex(masterBase)
		index = masterBase.Index
	}
	if index <= 0 {
		return fmt.Errorf("Device does not exist")
	}
	return h.LinkSetMasterByIndex(link, index)
}

// LinkSetNoMaster removes the master of the link device.
// Equivalent to: `ip link set $link nomaster`
func LinkSetNoMaster(link Link) error {
	return pkgHandle.LinkSetNoMaster(link)
}

// LinkSetNoMaster removes the master of the link device.
// Equivalent to: `ip link set $link nomaster`
func (h *Handle) LinkSetNoMaster(link Link) error {
	return h.LinkSetMasterByIndex(link, 0)
}

// LinkSetMasterByIndex sets the master of the link device.
// Equivalent to: `ip link set $link master $master`
func LinkSetMasterByIndex(link Link, masterIndex int) error {
	return pkgHandle.LinkSetMasterByIndex(link, masterIndex)
}

// LinkSetMasterByIndex sets the master of the link device.
// Equivalent to: `ip link set $link master $master`
func (h *Handle) LinkSetMasterByIndex(link Link, masterIndex int) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	b := make([]byte, 4)
	native.PutUint32(b, uint32(masterIndex))

	data := nl.NewRtAttr(syscall.IFLA_MASTER, b)
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetNsPid puts the device into a new network namespace. The
// pid must be a pid of a running process.
// Equivalent to: `ip link set $link netns $pid`
func LinkSetNsPid(link Link, nspid int) error {
	return pkgHandle.LinkSetNsPid(link, nspid)
}

// LinkSetNsPid puts the device into a new network namespace. The
// pid must be a pid of a running process.
// Equivalent to: `ip link set $link netns $pid`
func (h *Handle) LinkSetNsPid(link Link, nspid int) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	b := make([]byte, 4)
	native.PutUint32(b, uint32(nspid))

	data := nl.NewRtAttr(syscall.IFLA_NET_NS_PID, b)
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetNsFd puts the device into a new network namespace. The
// fd must be an open file descriptor to a network namespace.
// Similar to: `ip link set $link netns $ns`
func LinkSetNsFd(link Link, fd int) error {
	return pkgHandle.LinkSetNsFd(link, fd)
}

// LinkSetNsFd puts the device into a new network namespace. The
// fd must be an open file descriptor to a network namespace.
// Similar to: `ip link set $link netns $ns`
func (h *Handle) LinkSetNsFd(link Link, fd int) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	b := make([]byte, 4)
	native.PutUint32(b, uint32(fd))

	data := nl.NewRtAttr(nl.IFLA_NET_NS_FD, b)
	req.AddData(data)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// LinkSetXdpFd adds a bpf function to the driver. The fd must be a bpf
// program loaded with bpf(type=BPF_PROG_TYPE_XDP)
func LinkSetXdpFd(link Link, fd int) error {
	base := link.Attrs()
	ensureIndex(base)
	req := nl.NewNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	addXdpAttrs(&LinkXdp{Fd: fd}, req)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

func boolAttr(val bool) []byte {
	var v uint8
	if val {
		v = 1
	}
	return nl.Uint8Attr(v)
}

type vxlanPortRange struct {
	Lo, Hi uint16
}

func addVxlanAttrs(vxlan *Vxlan, linkInfo *nl.RtAttr) {
	data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
	nl.NewRtAttrChild(data, nl.IFLA_VXLAN_ID, nl.Uint32Attr(uint32(vxlan.VxlanId)))
	if vxlan.VtepDevIndex != 0 {
		nl.NewRtAttrChild(data, nl.IFLA_VXLAN_LINK, nl.Uint32Attr(uint32(vxlan.VtepDevIndex)))
	}
	if vxlan.SrcAddr != nil {
		ip := vxlan.SrcAddr.To4()
		if ip != nil {
			nl.NewRtAttrChild(data, nl.IFLA_VXLAN_LOCAL, []byte(ip))
		} else {
			ip = vxlan.SrcAddr.To16()
			if ip != nil {
				nl.NewRtAttrChild(data, nl.IFLA_VXLAN_LOCAL6, []byte(ip))
			}
		}
	}
	if vxlan.Group != nil {
		group := vxlan.Group.To4()
		if group != nil {
			nl.NewRtAttrChild(data, nl.IFLA_VXLAN_GROUP, []byte(group))
		} else {
			group = vxlan.Group.To16()
			if group != nil {
				nl.NewRtAttrChild(data, nl.IFLA_VXLAN_GROUP6, []byte(group))
			}
		}
	}

	nl.NewRtAttrChild(data, nl.IFLA_VXLAN_TTL, nl.Uint8Attr(uint8(vxlan.TTL)))
	nl.NewRtAttrChild(data, nl.IFLA_VXLAN_TOS, nl.Uint8Attr(uint8(vxlan.TOS)))
	nl.NewRtAttrChild(data, nl.IFLA_VXLAN_LEARNING, boolAttr(vxlan.Learning))
	nl.NewRtAttrChild(data, nl.IFLA_VXLAN_PROXY, boolAttr(vxlan.Proxy))
	nl.NewRtAttrChild(data, nl.IFLA_VXLAN_RSC, boolAttr(vxlan.RSC))
	nl.NewRtAttrChild(data, nl.IFLA_VXLAN_L2MISS, boolAttr(vxlan.L2miss))
	nl.NewRtAttrChild(data, nl.IFLA_VXLAN_L3MISS, boolAttr(vxlan.L3miss))

	if vxlan.UDPCSum {
		nl.NewRtAttrChild(data, nl.IFLA_VXLAN_UDP_CSUM, boolAttr(vxlan.UDPCSum))
	}
	if vxlan.GBP {
		nl.NewRtAttrChild(data, nl.IFLA_VXLAN_GBP, []byte{})
	}
	if vxlan.NoAge {
		nl.NewRtAttrChild(data, nl.IFLA_VXLAN_AGEING, nl.Uint32Attr(0))
	} else if vxlan.Age > 0 {
		nl.NewRtAttrChild(data, nl.IFLA_VXLAN_AGEING, nl.Uint32Attr(uint32(vxlan.Age)))
	}
	if vxlan.Limit > 0 {
		nl.NewRtAttrChild(data, nl.IFLA_VXLAN_LIMIT, nl.Uint32Attr(uint32(vxlan.Limit)))
	}
	if vxlan.Port > 0 {
		nl.NewRtAttrChild(data, nl.IFLA_VXLAN_PORT, htons(uint16(vxlan.Port)))
	}
	if vxlan.PortLow > 0 || vxlan.PortHigh > 0 {
		pr := vxlanPortRange{uint16(vxlan.PortLow), uint16(vxlan.PortHigh)}

		buf := new(bytes.Buffer)
		binary.Write(buf, binary.BigEndian, &pr)

		nl.NewRtAttrChild(data, nl.IFLA_VXLAN_PORT_RANGE, buf.Bytes())
	}
}

func addBondAttrs(bond *Bond, linkInfo *nl.RtAttr) {
	data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
	if bond.Mode >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_MODE, nl.Uint8Attr(uint8(bond.Mode)))
	}
	if bond.ActiveSlave >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_ACTIVE_SLAVE, nl.Uint32Attr(uint32(bond.ActiveSlave)))
	}
	if bond.Miimon >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_MIIMON, nl.Uint32Attr(uint32(bond.Miimon)))
	}
	if bond.UpDelay >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_UPDELAY, nl.Uint32Attr(uint32(bond.UpDelay)))
	}
	if bond.DownDelay >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_DOWNDELAY, nl.Uint32Attr(uint32(bond.DownDelay)))
	}
	if bond.UseCarrier >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_USE_CARRIER, nl.Uint8Attr(uint8(bond.UseCarrier)))
	}
	if bond.ArpInterval >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_ARP_INTERVAL, nl.Uint32Attr(uint32(bond.ArpInterval)))
	}
	if bond.ArpIpTargets != nil {
		msg := nl.NewRtAttrChild(data, nl.IFLA_BOND_ARP_IP_TARGET, nil)
		for i := range bond.ArpIpTargets {
			ip := bond.ArpIpTargets[i].To4()
			if ip != nil {
				nl.NewRtAttrChild(msg, i, []byte(ip))
				continue
			}
			ip = bond.ArpIpTargets[i].To16()
			if ip != nil {
				nl.NewRtAttrChild(msg, i, []byte(ip))
			}
		}
	}
	if bond.ArpValidate >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_ARP_VALIDATE, nl.Uint32Attr(uint32(bond.ArpValidate)))
	}
	if bond.ArpAllTargets >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_ARP_ALL_TARGETS, nl.Uint32Attr(uint32(bond.ArpAllTargets)))
	}
	if bond.Primary >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_PRIMARY, nl.Uint32Attr(uint32(bond.Primary)))
	}
	if bond.PrimaryReselect >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_PRIMARY_RESELECT, nl.Uint8Attr(uint8(bond.PrimaryReselect)))
	}
	if bond.FailOverMac >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_FAIL_OVER_MAC, nl.Uint8Attr(uint8(bond.FailOverMac)))
	}
	if bond.XmitHashPolicy >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_XMIT_HASH_POLICY, nl.Uint8Attr(uint8(bond.XmitHashPolicy)))
	}
	if bond.ResendIgmp >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_RESEND_IGMP, nl.Uint32Attr(uint32(bond.ResendIgmp)))
	}
	if bond.NumPeerNotif >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_NUM_PEER_NOTIF, nl.Uint8Attr(uint8(bond.NumPeerNotif)))
	}
	if bond.AllSlavesActive >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_ALL_SLAVES_ACTIVE, nl.Uint8Attr(uint8(bond.AllSlavesActive)))
	}
	if bond.MinLinks >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_MIN_LINKS, nl.Uint32Attr(uint32(bond.MinLinks)))
	}
	if bond.LpInterval >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_LP_INTERVAL, nl.Uint32Attr(uint32(bond.LpInterval)))
	}
	if bond.PackersPerSlave >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_PACKETS_PER_SLAVE, nl.Uint32Attr(uint32(bond.PackersPerSlave)))
	}
	if bond.LacpRate >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_AD_LACP_RATE, nl.Uint8Attr(uint8(bond.LacpRate)))
	}
	if bond.AdSelect >= 0 {
		nl.NewRtAttrChild(data, nl.IFLA_BOND_AD_SELECT, nl.Uint8Attr(uint8(bond.AdSelect)))
	}
}

// LinkAdd adds a new link device. The type and features of the device
// are taken from the parameters in the link object.
// Equivalent to: `ip link add $link`
func LinkAdd(link Link) error {
	return pkgHandle.LinkAdd(link)
}

// LinkAdd adds a new link device. The type and features of the device
// are taken fromt the parameters in the link object.
// Equivalent to: `ip link add $link`
func (h *Handle) LinkAdd(link Link) error {
	return h.linkModify(link, syscall.NLM_F_CREATE|syscall.NLM_F_EXCL|syscall.NLM_F_ACK)
}

func (h *Handle) linkModify(link Link, flags int) error {
	// TODO: set mtu and hardware address
	// TODO: support extra data for macvlan
	base := link.Attrs()

	if base.Name == "" {
		return fmt.Errorf("LinkAttrs.Name cannot be empty!")
	}

	if tuntap, ok := link.(*Tuntap); ok {
		// TODO: support user
		// TODO: support group
		// TODO: multi_queue
		// TODO: support non- persistent
		if tuntap.Mode < syscall.IFF_TUN || tuntap.Mode > syscall.IFF_TAP {
			return fmt.Errorf("Tuntap.Mode %v unknown!", tuntap.Mode)
		}
		file, err := os.OpenFile("/dev/net/tun", os.O_RDWR, 0)
		if err != nil {
			return err
		}
		defer file.Close()
		var req ifReq
		if tuntap.Flags == 0 {
			req.Flags = uint16(TUNTAP_DEFAULTS)
		} else {
			req.Flags = uint16(tuntap.Flags)
		}
		req.Flags |= uint16(tuntap.Mode)
		copy(req.Name[:15], base.Name)
		_, _, errno := syscall.Syscall(syscall.SYS_IOCTL, file.Fd(), uintptr(syscall.TUNSETIFF), uintptr(unsafe.Pointer(&req)))
		if errno != 0 {
			return fmt.Errorf("Tuntap IOCTL TUNSETIFF failed, errno %v", errno)
		}
		_, _, errno = syscall.Syscall(syscall.SYS_IOCTL, file.Fd(), uintptr(syscall.TUNSETPERSIST), 1)
		if errno != 0 {
			return fmt.Errorf("Tuntap IOCTL TUNSETPERSIST failed, errno %v", errno)
		}
		h.ensureIndex(base)

		// can't set master during create, so set it afterwards
		if base.MasterIndex != 0 {
			// TODO: verify MasterIndex is actually a bridge?
			return h.LinkSetMasterByIndex(link, base.MasterIndex)
		}
		return nil
	}

	req := h.newNetlinkRequest(syscall.RTM_NEWLINK, flags)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	// TODO: make it shorter
	if base.Flags&net.FlagUp != 0 {
		msg.Change = syscall.IFF_UP
		msg.Flags = syscall.IFF_UP
	}
	if base.Flags&net.FlagBroadcast != 0 {
		msg.Change |= syscall.IFF_BROADCAST
		msg.Flags |= syscall.IFF_BROADCAST
	}
	if base.Flags&net.FlagLoopback != 0 {
		msg.Change |= syscall.IFF_LOOPBACK
		msg.Flags |= syscall.IFF_LOOPBACK
	}
	if base.Flags&net.FlagPointToPoint != 0 {
		msg.Change |= syscall.IFF_POINTOPOINT
		msg.Flags |= syscall.IFF_POINTOPOINT
	}
	if base.Flags&net.FlagMulticast != 0 {
		msg.Change |= syscall.IFF_MULTICAST
		msg.Flags |= syscall.IFF_MULTICAST
	}
	req.AddData(msg)

	if base.ParentIndex != 0 {
		b := make([]byte, 4)
		native.PutUint32(b, uint32(base.ParentIndex))
		data := nl.NewRtAttr(syscall.IFLA_LINK, b)
		req.AddData(data)
	} else if link.Type() == "ipvlan" {
		return fmt.Errorf("Can't create ipvlan link without ParentIndex")
	}

	nameData := nl.NewRtAttr(syscall.IFLA_IFNAME, nl.ZeroTerminated(base.Name))
	req.AddData(nameData)

	if base.MTU > 0 {
		mtu := nl.NewRtAttr(syscall.IFLA_MTU, nl.Uint32Attr(uint32(base.MTU)))
		req.AddData(mtu)
	}

	if base.TxQLen >= 0 {
		qlen := nl.NewRtAttr(syscall.IFLA_TXQLEN, nl.Uint32Attr(uint32(base.TxQLen)))
		req.AddData(qlen)
	}

	if base.Namespace != nil {
		var attr *nl.RtAttr
		switch base.Namespace.(type) {
		case NsPid:
			val := nl.Uint32Attr(uint32(base.Namespace.(NsPid)))
			attr = nl.NewRtAttr(syscall.IFLA_NET_NS_PID, val)
		case NsFd:
			val := nl.Uint32Attr(uint32(base.Namespace.(NsFd)))
			attr = nl.NewRtAttr(nl.IFLA_NET_NS_FD, val)
		}

		req.AddData(attr)
	}

	if base.Xdp != nil {
		addXdpAttrs(base.Xdp, req)
	}

	linkInfo := nl.NewRtAttr(syscall.IFLA_LINKINFO, nil)
	nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_KIND, nl.NonZeroTerminated(link.Type()))

	if vlan, ok := link.(*Vlan); ok {
		b := make([]byte, 2)
		native.PutUint16(b, uint16(vlan.VlanId))
		data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
		nl.NewRtAttrChild(data, nl.IFLA_VLAN_ID, b)
	} else if veth, ok := link.(*Veth); ok {
		data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
		peer := nl.NewRtAttrChild(data, nl.VETH_INFO_PEER, nil)
		nl.NewIfInfomsgChild(peer, syscall.AF_UNSPEC)
		nl.NewRtAttrChild(peer, syscall.IFLA_IFNAME, nl.ZeroTerminated(veth.PeerName))
		if base.TxQLen >= 0 {
			nl.NewRtAttrChild(peer, syscall.IFLA_TXQLEN, nl.Uint32Attr(uint32(base.TxQLen)))
		}
		if base.MTU > 0 {
			nl.NewRtAttrChild(peer, syscall.IFLA_MTU, nl.Uint32Attr(uint32(base.MTU)))
		}

	} else if vxlan, ok := link.(*Vxlan); ok {
		addVxlanAttrs(vxlan, linkInfo)
	} else if bond, ok := link.(*Bond); ok {
		addBondAttrs(bond, linkInfo)
	} else if ipv, ok := link.(*IPVlan); ok {
		data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
		nl.NewRtAttrChild(data, nl.IFLA_IPVLAN_MODE, nl.Uint16Attr(uint16(ipv.Mode)))
	} else if macv, ok := link.(*Macvlan); ok {
		if macv.Mode != MACVLAN_MODE_DEFAULT {
			data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
			nl.NewRtAttrChild(data, nl.IFLA_MACVLAN_MODE, nl.Uint32Attr(macvlanModes[macv.Mode]))
		}
	} else if macv, ok := link.(*Macvtap); ok {
		if macv.Mode != MACVLAN_MODE_DEFAULT {
			data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
			nl.NewRtAttrChild(data, nl.IFLA_MACVLAN_MODE, nl.Uint32Attr(macvlanModes[macv.Mode]))
		}
	} else if gretap, ok := link.(*Gretap); ok {
		addGretapAttrs(gretap, linkInfo)
	} else if iptun, ok := link.(*Iptun); ok {
		addIptunAttrs(iptun, linkInfo)
	} else if vti, ok := link.(*Vti); ok {
		addVtiAttrs(vti, linkInfo)
	} else if vrf, ok := link.(*Vrf); ok {
		addVrfAttrs(vrf, linkInfo)
	} else if bridge, ok := link.(*Bridge); ok {
		addBridgeAttrs(bridge, linkInfo)
	} else if gtp, ok := link.(*GTP); ok {
		addGTPAttrs(gtp, linkInfo)
	}

	req.AddData(linkInfo)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	if err != nil {
		return err
	}

	h.ensureIndex(base)

	// can't set master during create, so set it afterwards
	if base.MasterIndex != 0 {
		// TODO: verify MasterIndex is actually a bridge?
		return h.LinkSetMasterByIndex(link, base.MasterIndex)
	}
	return nil
}

// LinkDel deletes link device. Either Index or Name must be set in
// the link object for it to be deleted. The other values are ignored.
// Equivalent to: `ip link del $link`
func LinkDel(link Link) error {
	return pkgHandle.LinkDel(link)
}

// LinkDel deletes link device. Either Index or Name must be set in
// the link object for it to be deleted. The other values are ignored.
// Equivalent to: `ip link del $link`
func (h *Handle) LinkDel(link Link) error {
	base := link.Attrs()

	h.ensureIndex(base)

	req := h.newNetlinkRequest(syscall.RTM_DELLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

func (h *Handle) linkByNameDump(name string) (Link, error) {
	links, err := h.LinkList()
	if err != nil {
		return nil, err
	}

	for _, link := range links {
		if link.Attrs().Name == name {
			return link, nil
		}
	}
	return nil, LinkNotFoundError{fmt.Errorf("Link %s not found", name)}
}

func (h *Handle) linkByAliasDump(alias string) (Link, error) {
	links, err := h.LinkList()
	if err != nil {
		return nil, err
	}

	for _, link := range links {
		if link.Attrs().Alias == alias {
			return link, nil
		}
	}
	return nil, LinkNotFoundError{fmt.Errorf("Link alias %s not found", alias)}
}

// LinkByName finds a link by name and returns a pointer to the object.
func LinkByName(name string) (Link, error) {
	return pkgHandle.LinkByName(name)
}

// LinkByName finds a link by name and returns a pointer to the object.
func (h *Handle) LinkByName(name string) (Link, error) {
	if h.lookupByDump {
		return h.linkByNameDump(name)
	}

	req := h.newNetlinkRequest(syscall.RTM_GETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	req.AddData(msg)

	nameData := nl.NewRtAttr(syscall.IFLA_IFNAME, nl.ZeroTerminated(name))
	req.AddData(nameData)

	link, err := execGetLink(req)
	if err == syscall.EINVAL {
		// older kernels don't support looking up via IFLA_IFNAME
		// so fall back to dumping all links
		h.lookupByDump = true
		return h.linkByNameDump(name)
	}

	return link, err
}

// LinkByAlias finds a link by its alias and returns a pointer to the object.
// If there are multiple links with the alias it returns the first one
func LinkByAlias(alias string) (Link, error) {
	return pkgHandle.LinkByAlias(alias)
}

// LinkByAlias finds a link by its alias and returns a pointer to the object.
// If there are multiple links with the alias it returns the first one
func (h *Handle) LinkByAlias(alias string) (Link, error) {
	if h.lookupByDump {
		return h.linkByAliasDump(alias)
	}

	req := h.newNetlinkRequest(syscall.RTM_GETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	req.AddData(msg)

	nameData := nl.NewRtAttr(syscall.IFLA_IFALIAS, nl.ZeroTerminated(alias))
	req.AddData(nameData)

	link, err := execGetLink(req)
	if err == syscall.EINVAL {
		// older kernels don't support looking up via IFLA_IFALIAS
		// so fall back to dumping all links
		h.lookupByDump = true
		return h.linkByAliasDump(alias)
	}

	return link, err
}

// LinkByIndex finds a link by index and returns a pointer to the object.
func LinkByIndex(index int) (Link, error) {
	return pkgHandle.LinkByIndex(index)
}

// LinkByIndex finds a link by index and returns a pointer to the object.
func (h *Handle) LinkByIndex(index int) (Link, error) {
	req := h.newNetlinkRequest(syscall.RTM_GETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	msg.Index = int32(index)
	req.AddData(msg)

	return execGetLink(req)
}

func execGetLink(req *nl.NetlinkRequest) (Link, error) {
	msgs, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	if err != nil {
		if errno, ok := err.(syscall.Errno); ok {
			if errno == syscall.ENODEV {
				return nil, LinkNotFoundError{fmt.Errorf("Link not found")}
			}
		}
		return nil, err
	}

	switch {
	case len(msgs) == 0:
		return nil, LinkNotFoundError{fmt.Errorf("Link not found")}

	case len(msgs) == 1:
		return LinkDeserialize(nil, msgs[0])

	default:
		return nil, fmt.Errorf("More than one link found")
	}
}

// linkDeserialize deserializes a raw message received from netlink into
// a link object.
func LinkDeserialize(hdr *syscall.NlMsghdr, m []byte) (Link, error) {
	msg := nl.DeserializeIfInfomsg(m)

	attrs, err := nl.ParseRouteAttr(m[msg.Len():])
	if err != nil {
		return nil, err
	}

	base := LinkAttrs{Index: int(msg.Index), RawFlags: msg.Flags, Flags: linkFlags(msg.Flags), EncapType: msg.EncapType()}
	if msg.Flags&syscall.IFF_PROMISC != 0 {
		base.Promisc = 1
	}
	var (
		link     Link
		stats32  []byte
		stats64  []byte
		linkType string
	)
	for _, attr := range attrs {
		switch attr.Attr.Type {
		case syscall.IFLA_LINKINFO:
			infos, err := nl.ParseRouteAttr(attr.Value)
			if err != nil {
				return nil, err
			}
			for _, info := range infos {
				switch info.Attr.Type {
				case nl.IFLA_INFO_KIND:
					linkType = string(info.Value[:len(info.Value)-1])
					switch linkType {
					case "dummy":
						link = &Dummy{}
					case "ifb":
						link = &Ifb{}
					case "bridge":
						link = &Bridge{}
					case "vlan":
						link = &Vlan{}
					case "veth":
						link = &Veth{}
					case "vxlan":
						link = &Vxlan{}
					case "bond":
						link = &Bond{}
					case "ipvlan":
						link = &IPVlan{}
					case "macvlan":
						link = &Macvlan{}
					case "macvtap":
						link = &Macvtap{}
					case "gretap":
						link = &Gretap{}
					case "ipip":
						link = &Iptun{}
					case "vti":
						link = &Vti{}
					case "vrf":
						link = &Vrf{}
					case "gtp":
						link = &GTP{}
					default:
						link = &GenericLink{LinkType: linkType}
					}
				case nl.IFLA_INFO_DATA:
					data, err := nl.ParseRouteAttr(info.Value)
					if err != nil {
						return nil, err
					}
					switch linkType {
					case "vlan":
						parseVlanData(link, data)
					case "vxlan":
						parseVxlanData(link, data)
					case "bond":
						parseBondData(link, data)
					case "ipvlan":
						parseIPVlanData(link, data)
					case "macvlan":
						parseMacvlanData(link, data)
					case "macvtap":
						parseMacvtapData(link, data)
					case "gretap":
						parseGretapData(link, data)
					case "ipip":
						parseIptunData(link, data)
					case "vti":
						parseVtiData(link, data)
					case "vrf":
						parseVrfData(link, data)
					case "bridge":
						parseBridgeData(link, data)
					case "gtp":
						parseGTPData(link, data)
					}
				}
			}
		case syscall.IFLA_ADDRESS:
			var nonzero bool
			for _, b := range attr.Value {
				if b != 0 {
					nonzero = true
				}
			}
			if nonzero {
				base.HardwareAddr = attr.Value[:]
			}
		case syscall.IFLA_IFNAME:
			base.Name = string(attr.Value[:len(attr.Value)-1])
		case syscall.IFLA_MTU:
			base.MTU = int(native.Uint32(attr.Value[0:4]))
		case syscall.IFLA_LINK:
			base.ParentIndex = int(native.Uint32(attr.Value[0:4]))
		case syscall.IFLA_MASTER:
			base.MasterIndex = int(native.Uint32(attr.Value[0:4]))
		case syscall.IFLA_TXQLEN:
			base.TxQLen = int(native.Uint32(attr.Value[0:4]))
		case syscall.IFLA_IFALIAS:
			base.Alias = string(attr.Value[:len(attr.Value)-1])
		case syscall.IFLA_STATS:
			stats32 = attr.Value[:]
		case IFLA_STATS64:
			stats64 = attr.Value[:]
		case nl.IFLA_XDP:
			xdp, err := parseLinkXdp(attr.Value[:])
			if err != nil {
				return nil, err
			}
			base.Xdp = xdp
		case syscall.IFLA_PROTINFO | syscall.NLA_F_NESTED:
			if hdr != nil && hdr.Type == syscall.RTM_NEWLINK &&
				msg.Family == syscall.AF_BRIDGE {
				attrs, err := nl.ParseRouteAttr(attr.Value[:])
				if err != nil {
					return nil, err
				}
				base.Protinfo = parseProtinfo(attrs)
			}
		case syscall.IFLA_OPERSTATE:
			base.OperState = LinkOperState(uint8(attr.Value[0]))
		}
	}

	if stats64 != nil {
		base.Statistics = parseLinkStats64(stats64)
	} else if stats32 != nil {
		base.Statistics = parseLinkStats32(stats32)
	}

	// Links that don't have IFLA_INFO_KIND are hardware devices
	if link == nil {
		link = &Device{}
	}
	*link.Attrs() = base

	return link, nil
}

// LinkList gets a list of link devices.
// Equivalent to: `ip link show`
func LinkList() ([]Link, error) {
	return pkgHandle.LinkList()
}

// LinkList gets a list of link devices.
// Equivalent to: `ip link show`
func (h *Handle) LinkList() ([]Link, error) {
	// NOTE(vish): This duplicates functionality in net/iface_linux.go, but we need
	//             to get the message ourselves to parse link type.
	req := h.newNetlinkRequest(syscall.RTM_GETLINK, syscall.NLM_F_DUMP)

	msg := nl.NewIfInfomsg(syscall.AF_UNSPEC)
	req.AddData(msg)

	msgs, err := req.Execute(syscall.NETLINK_ROUTE, syscall.RTM_NEWLINK)
	if err != nil {
		return nil, err
	}

	var res []Link
	for _, m := range msgs {
		link, err := LinkDeserialize(nil, m)
		if err != nil {
			return nil, err
		}
		res = append(res, link)
	}

	return res, nil
}

// LinkUpdate is used to pass information back from LinkSubscribe()
type LinkUpdate struct {
	nl.IfInfomsg
	Header syscall.NlMsghdr
	Link
}

// LinkSubscribe takes a chan down which notifications will be sent
// when links change.  Close the 'done' chan to stop subscription.
func LinkSubscribe(ch chan<- LinkUpdate, done <-chan struct{}) error {
	return linkSubscribe(netns.None(), netns.None(), ch, done)
}

// LinkSubscribeAt works like LinkSubscribe plus it allows the caller
// to choose the network namespace in which to subscribe (ns).
func LinkSubscribeAt(ns netns.NsHandle, ch chan<- LinkUpdate, done <-chan struct{}) error {
	return linkSubscribe(ns, netns.None(), ch, done)
}

func linkSubscribe(newNs, curNs netns.NsHandle, ch chan<- LinkUpdate, done <-chan struct{}) error {
	s, err := nl.SubscribeAt(newNs, curNs, syscall.NETLINK_ROUTE, syscall.RTNLGRP_LINK)
	if err != nil {
		return err
	}
	if done != nil {
		go func() {
			<-done
			s.Close()
		}()
	}
	go func() {
		defer close(ch)
		for {
			msgs, err := s.Receive()
			if err != nil {
				return
			}
			for _, m := range msgs {
				ifmsg := nl.DeserializeIfInfomsg(m.Data)
				link, err := LinkDeserialize(&m.Header, m.Data)
				if err != nil {
					return
				}
				ch <- LinkUpdate{IfInfomsg: *ifmsg, Header: m.Header, Link: link}
			}
		}
	}()

	return nil
}

func LinkSetHairpin(link Link, mode bool) error {
	return pkgHandle.LinkSetHairpin(link, mode)
}

func (h *Handle) LinkSetHairpin(link Link, mode bool) error {
	return h.setProtinfoAttr(link, mode, nl.IFLA_BRPORT_MODE)
}

func LinkSetGuard(link Link, mode bool) error {
	return pkgHandle.LinkSetGuard(link, mode)
}

func (h *Handle) LinkSetGuard(link Link, mode bool) error {
	return h.setProtinfoAttr(link, mode, nl.IFLA_BRPORT_GUARD)
}

func LinkSetFastLeave(link Link, mode bool) error {
	return pkgHandle.LinkSetFastLeave(link, mode)
}

func (h *Handle) LinkSetFastLeave(link Link, mode bool) error {
	return h.setProtinfoAttr(link, mode, nl.IFLA_BRPORT_FAST_LEAVE)
}

func LinkSetLearning(link Link, mode bool) error {
	return pkgHandle.LinkSetLearning(link, mode)
}

func (h *Handle) LinkSetLearning(link Link, mode bool) error {
	return h.setProtinfoAttr(link, mode, nl.IFLA_BRPORT_LEARNING)
}

func LinkSetRootBlock(link Link, mode bool) error {
	return pkgHandle.LinkSetRootBlock(link, mode)
}

func (h *Handle) LinkSetRootBlock(link Link, mode bool) error {
	return h.setProtinfoAttr(link, mode, nl.IFLA_BRPORT_PROTECT)
}

func LinkSetFlood(link Link, mode bool) error {
	return pkgHandle.LinkSetFlood(link, mode)
}

func (h *Handle) LinkSetFlood(link Link, mode bool) error {
	return h.setProtinfoAttr(link, mode, nl.IFLA_BRPORT_UNICAST_FLOOD)
}

func LinkSetBrProxyArp(link Link, mode bool) error {
	return pkgHandle.LinkSetBrProxyArp(link, mode)
}

func (h *Handle) LinkSetBrProxyArp(link Link, mode bool) error {
	return h.setProtinfoAttr(link, mode, nl.IFLA_BRPORT_PROXYARP)
}

func LinkSetBrProxyArpWiFi(link Link, mode bool) error {
	return pkgHandle.LinkSetBrProxyArpWiFi(link, mode)
}

func (h *Handle) LinkSetBrProxyArpWiFi(link Link, mode bool) error {
	return h.setProtinfoAttr(link, mode, nl.IFLA_BRPORT_PROXYARP_WIFI)
}

func (h *Handle) setProtinfoAttr(link Link, mode bool, attr int) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(syscall.RTM_SETLINK, syscall.NLM_F_ACK)

	msg := nl.NewIfInfomsg(syscall.AF_BRIDGE)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	br := nl.NewRtAttr(syscall.IFLA_PROTINFO|syscall.NLA_F_NESTED, nil)
	nl.NewRtAttrChild(br, attr, boolToByte(mode))
	req.AddData(br)
	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	if err != nil {
		return err
	}
	return nil
}

func parseVlanData(link Link, data []syscall.NetlinkRouteAttr) {
	vlan := link.(*Vlan)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.IFLA_VLAN_ID:
			vlan.VlanId = int(native.Uint16(datum.Value[0:2]))
		}
	}
}

func parseVxlanData(link Link, data []syscall.NetlinkRouteAttr) {
	vxlan := link.(*Vxlan)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.IFLA_VXLAN_ID:
			vxlan.VxlanId = int(native.Uint32(datum.Value[0:4]))
		case nl.IFLA_VXLAN_LINK:
			vxlan.VtepDevIndex = int(native.Uint32(datum.Value[0:4]))
		case nl.IFLA_VXLAN_LOCAL:
			vxlan.SrcAddr = net.IP(datum.Value[0:4])
		case nl.IFLA_VXLAN_LOCAL6:
			vxlan.SrcAddr = net.IP(datum.Value[0:16])
		case nl.IFLA_VXLAN_GROUP:
			vxlan.Group = net.IP(datum.Value[0:4])
		case nl.IFLA_VXLAN_GROUP6:
			vxlan.Group = net.IP(datum.Value[0:16])
		case nl.IFLA_VXLAN_TTL:
			vxlan.TTL = int(datum.Value[0])
		case nl.IFLA_VXLAN_TOS:
			vxlan.TOS = int(datum.Value[0])
		case nl.IFLA_VXLAN_LEARNING:
			vxlan.Learning = int8(datum.Value[0]) != 0
		case nl.IFLA_VXLAN_PROXY:
			vxlan.Proxy = int8(datum.Value[0]) != 0
		case nl.IFLA_VXLAN_RSC:
			vxlan.RSC = int8(datum.Value[0]) != 0
		case nl.IFLA_VXLAN_L2MISS:
			vxlan.L2miss = int8(datum.Value[0]) != 0
		case nl.IFLA_VXLAN_L3MISS:
			vxlan.L3miss = int8(datum.Value[0]) != 0
		case nl.IFLA_VXLAN_UDP_CSUM:
			vxlan.UDPCSum = int8(datum.Value[0]) != 0
		case nl.IFLA_VXLAN_GBP:
			vxlan.GBP = true
		case nl.IFLA_VXLAN_AGEING:
			vxlan.Age = int(native.Uint32(datum.Value[0:4]))
			vxlan.NoAge = vxlan.Age == 0
		case nl.IFLA_VXLAN_LIMIT:
			vxlan.Limit = int(native.Uint32(datum.Value[0:4]))
		case nl.IFLA_VXLAN_PORT:
			vxlan.Port = int(ntohs(datum.Value[0:2]))
		case nl.IFLA_VXLAN_PORT_RANGE:
			buf := bytes.NewBuffer(datum.Value[0:4])
			var pr vxlanPortRange
			if binary.Read(buf, binary.BigEndian, &pr) != nil {
				vxlan.PortLow = int(pr.Lo)
				vxlan.PortHigh = int(pr.Hi)
			}
		}
	}
}

func parseBondData(link Link, data []syscall.NetlinkRouteAttr) {
	bond := link.(*Bond)
	for i := range data {
		switch data[i].Attr.Type {
		case nl.IFLA_BOND_MODE:
			bond.Mode = BondMode(data[i].Value[0])
		case nl.IFLA_BOND_ACTIVE_SLAVE:
			bond.ActiveSlave = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_MIIMON:
			bond.Miimon = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_UPDELAY:
			bond.UpDelay = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_DOWNDELAY:
			bond.DownDelay = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_USE_CARRIER:
			bond.UseCarrier = int(data[i].Value[0])
		case nl.IFLA_BOND_ARP_INTERVAL:
			bond.ArpInterval = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_ARP_IP_TARGET:
			// TODO: implement
		case nl.IFLA_BOND_ARP_VALIDATE:
			bond.ArpValidate = BondArpValidate(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_ARP_ALL_TARGETS:
			bond.ArpAllTargets = BondArpAllTargets(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_PRIMARY:
			bond.Primary = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_PRIMARY_RESELECT:
			bond.PrimaryReselect = BondPrimaryReselect(data[i].Value[0])
		case nl.IFLA_BOND_FAIL_OVER_MAC:
			bond.FailOverMac = BondFailOverMac(data[i].Value[0])
		case nl.IFLA_BOND_XMIT_HASH_POLICY:
			bond.XmitHashPolicy = BondXmitHashPolicy(data[i].Value[0])
		case nl.IFLA_BOND_RESEND_IGMP:
			bond.ResendIgmp = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_NUM_PEER_NOTIF:
			bond.NumPeerNotif = int(data[i].Value[0])
		case nl.IFLA_BOND_ALL_SLAVES_ACTIVE:
			bond.AllSlavesActive = int(data[i].Value[0])
		case nl.IFLA_BOND_MIN_LINKS:
			bond.MinLinks = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_LP_INTERVAL:
			bond.LpInterval = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_PACKETS_PER_SLAVE:
			bond.PackersPerSlave = int(native.Uint32(data[i].Value[0:4]))
		case nl.IFLA_BOND_AD_LACP_RATE:
			bond.LacpRate = BondLacpRate(data[i].Value[0])
		case nl.IFLA_BOND_AD_SELECT:
			bond.AdSelect = BondAdSelect(data[i].Value[0])
		case nl.IFLA_BOND_AD_INFO:
			// TODO: implement
		}
	}
}

func parseIPVlanData(link Link, data []syscall.NetlinkRouteAttr) {
	ipv := link.(*IPVlan)
	for _, datum := range data {
		if datum.Attr.Type == nl.IFLA_IPVLAN_MODE {
			ipv.Mode = IPVlanMode(native.Uint32(datum.Value[0:4]))
			return
		}
	}
}

func parseMacvtapData(link Link, data []syscall.NetlinkRouteAttr) {
	macv := link.(*Macvtap)
	parseMacvlanData(&macv.Macvlan, data)
}

func parseMacvlanData(link Link, data []syscall.NetlinkRouteAttr) {
	macv := link.(*Macvlan)
	for _, datum := range data {
		if datum.Attr.Type == nl.IFLA_MACVLAN_MODE {
			switch native.Uint32(datum.Value[0:4]) {
			case nl.MACVLAN_MODE_PRIVATE:
				macv.Mode = MACVLAN_MODE_PRIVATE
			case nl.MACVLAN_MODE_VEPA:
				macv.Mode = MACVLAN_MODE_VEPA
			case nl.MACVLAN_MODE_BRIDGE:
				macv.Mode = MACVLAN_MODE_BRIDGE
			case nl.MACVLAN_MODE_PASSTHRU:
				macv.Mode = MACVLAN_MODE_PASSTHRU
			case nl.MACVLAN_MODE_SOURCE:
				macv.Mode = MACVLAN_MODE_SOURCE
			}
			return
		}
	}
}

// copied from pkg/net_linux.go
func linkFlags(rawFlags uint32) net.Flags {
	var f net.Flags
	if rawFlags&syscall.IFF_UP != 0 {
		f |= net.FlagUp
	}
	if rawFlags&syscall.IFF_BROADCAST != 0 {
		f |= net.FlagBroadcast
	}
	if rawFlags&syscall.IFF_LOOPBACK != 0 {
		f |= net.FlagLoopback
	}
	if rawFlags&syscall.IFF_POINTOPOINT != 0 {
		f |= net.FlagPointToPoint
	}
	if rawFlags&syscall.IFF_MULTICAST != 0 {
		f |= net.FlagMulticast
	}
	return f
}

func addGretapAttrs(gretap *Gretap, linkInfo *nl.RtAttr) {
	data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)

	ip := gretap.Local.To4()
	if ip != nil {
		nl.NewRtAttrChild(data, nl.IFLA_GRE_LOCAL, []byte(ip))
	}
	ip = gretap.Remote.To4()
	if ip != nil {
		nl.NewRtAttrChild(data, nl.IFLA_GRE_REMOTE, []byte(ip))
	}

	if gretap.IKey != 0 {
		nl.NewRtAttrChild(data, nl.IFLA_GRE_IKEY, htonl(gretap.IKey))
		gretap.IFlags |= uint16(nl.GRE_KEY)
	}

	if gretap.OKey != 0 {
		nl.NewRtAttrChild(data, nl.IFLA_GRE_OKEY, htonl(gretap.OKey))
		gretap.OFlags |= uint16(nl.GRE_KEY)
	}

	nl.NewRtAttrChild(data, nl.IFLA_GRE_IFLAGS, htons(gretap.IFlags))
	nl.NewRtAttrChild(data, nl.IFLA_GRE_OFLAGS, htons(gretap.OFlags))

	if gretap.Link != 0 {
		nl.NewRtAttrChild(data, nl.IFLA_GRE_LINK, nl.Uint32Attr(gretap.Link))
	}

	nl.NewRtAttrChild(data, nl.IFLA_GRE_PMTUDISC, nl.Uint8Attr(gretap.PMtuDisc))
	nl.NewRtAttrChild(data, nl.IFLA_GRE_TTL, nl.Uint8Attr(gretap.Ttl))
	nl.NewRtAttrChild(data, nl.IFLA_GRE_TOS, nl.Uint8Attr(gretap.Tos))
	nl.NewRtAttrChild(data, nl.IFLA_GRE_ENCAP_TYPE, nl.Uint16Attr(gretap.EncapType))
	nl.NewRtAttrChild(data, nl.IFLA_GRE_ENCAP_FLAGS, nl.Uint16Attr(gretap.EncapFlags))
	nl.NewRtAttrChild(data, nl.IFLA_GRE_ENCAP_SPORT, htons(gretap.EncapSport))
	nl.NewRtAttrChild(data, nl.IFLA_GRE_ENCAP_DPORT, htons(gretap.EncapDport))
}

func parseGretapData(link Link, data []syscall.NetlinkRouteAttr) {
	gre := link.(*Gretap)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.IFLA_GRE_OKEY:
			gre.IKey = ntohl(datum.Value[0:4])
		case nl.IFLA_GRE_IKEY:
			gre.OKey = ntohl(datum.Value[0:4])
		case nl.IFLA_GRE_LOCAL:
			gre.Local = net.IP(datum.Value[0:4])
		case nl.IFLA_GRE_REMOTE:
			gre.Remote = net.IP(datum.Value[0:4])
		case nl.IFLA_GRE_ENCAP_SPORT:
			gre.EncapSport = ntohs(datum.Value[0:2])
		case nl.IFLA_GRE_ENCAP_DPORT:
			gre.EncapDport = ntohs(datum.Value[0:2])
		case nl.IFLA_GRE_IFLAGS:
			gre.IFlags = ntohs(datum.Value[0:2])
		case nl.IFLA_GRE_OFLAGS:
			gre.OFlags = ntohs(datum.Value[0:2])

		case nl.IFLA_GRE_TTL:
			gre.Ttl = uint8(datum.Value[0])
		case nl.IFLA_GRE_TOS:
			gre.Tos = uint8(datum.Value[0])
		case nl.IFLA_GRE_PMTUDISC:
			gre.PMtuDisc = uint8(datum.Value[0])
		case nl.IFLA_GRE_ENCAP_TYPE:
			gre.EncapType = native.Uint16(datum.Value[0:2])
		case nl.IFLA_GRE_ENCAP_FLAGS:
			gre.EncapFlags = native.Uint16(datum.Value[0:2])
		}
	}
}

func parseLinkStats32(data []byte) *LinkStatistics {
	return (*LinkStatistics)((*LinkStatistics32)(unsafe.Pointer(&data[0:SizeofLinkStats32][0])).to64())
}

func parseLinkStats64(data []byte) *LinkStatistics {
	return (*LinkStatistics)((*LinkStatistics64)(unsafe.Pointer(&data[0:SizeofLinkStats64][0])))
}

func addXdpAttrs(xdp *LinkXdp, req *nl.NetlinkRequest) {
	attrs := nl.NewRtAttr(nl.IFLA_XDP|syscall.NLA_F_NESTED, nil)
	b := make([]byte, 4)
	native.PutUint32(b, uint32(xdp.Fd))
	nl.NewRtAttrChild(attrs, nl.IFLA_XDP_FD, b)
	native.PutUint32(b, xdp.Flags)
	nl.NewRtAttrChild(attrs, nl.IFLA_XDP_FLAGS, b)
	req.AddData(attrs)
}

func parseLinkXdp(data []byte) (*LinkXdp, error) {
	attrs, err := nl.ParseRouteAttr(data)
	if err != nil {
		return nil, err
	}
	xdp := &LinkXdp{}
	for _, attr := range attrs {
		switch attr.Attr.Type {
		case nl.IFLA_XDP_FD:
			xdp.Fd = int(native.Uint32(attr.Value[0:4]))
		case nl.IFLA_XDP_ATTACHED:
			xdp.Attached = attr.Value[0] != 0
		case nl.IFLA_XDP_FLAGS:
			xdp.Flags = native.Uint32(attr.Value[0:4])
		}
	}
	return xdp, nil
}

func addIptunAttrs(iptun *Iptun, linkInfo *nl.RtAttr) {
	data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)

	ip := iptun.Local.To4()
	if ip != nil {
		nl.NewRtAttrChild(data, nl.IFLA_IPTUN_LOCAL, []byte(ip))
	}

	ip = iptun.Remote.To4()
	if ip != nil {
		nl.NewRtAttrChild(data, nl.IFLA_IPTUN_REMOTE, []byte(ip))
	}

	if iptun.Link != 0 {
		nl.NewRtAttrChild(data, nl.IFLA_IPTUN_LINK, nl.Uint32Attr(iptun.Link))
	}
	nl.NewRtAttrChild(data, nl.IFLA_IPTUN_PMTUDISC, nl.Uint8Attr(iptun.PMtuDisc))
	nl.NewRtAttrChild(data, nl.IFLA_IPTUN_TTL, nl.Uint8Attr(iptun.Ttl))
	nl.NewRtAttrChild(data, nl.IFLA_IPTUN_TOS, nl.Uint8Attr(iptun.Tos))
}

func parseIptunData(link Link, data []syscall.NetlinkRouteAttr) {
	iptun := link.(*Iptun)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.IFLA_IPTUN_LOCAL:
			iptun.Local = net.IP(datum.Value[0:4])
		case nl.IFLA_IPTUN_REMOTE:
			iptun.Remote = net.IP(datum.Value[0:4])
		case nl.IFLA_IPTUN_TTL:
			iptun.Ttl = uint8(datum.Value[0])
		case nl.IFLA_IPTUN_TOS:
			iptun.Tos = uint8(datum.Value[0])
		case nl.IFLA_IPTUN_PMTUDISC:
			iptun.PMtuDisc = uint8(datum.Value[0])
		}
	}
}

func addVtiAttrs(vti *Vti, linkInfo *nl.RtAttr) {
	data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)

	ip := vti.Local.To4()
	if ip != nil {
		nl.NewRtAttrChild(data, nl.IFLA_VTI_LOCAL, []byte(ip))
	}

	ip = vti.Remote.To4()
	if ip != nil {
		nl.NewRtAttrChild(data, nl.IFLA_VTI_REMOTE, []byte(ip))
	}

	if vti.Link != 0 {
		nl.NewRtAttrChild(data, nl.IFLA_VTI_LINK, nl.Uint32Attr(vti.Link))
	}

	nl.NewRtAttrChild(data, nl.IFLA_VTI_IKEY, htonl(vti.IKey))
	nl.NewRtAttrChild(data, nl.IFLA_VTI_OKEY, htonl(vti.OKey))
}

func parseVtiData(link Link, data []syscall.NetlinkRouteAttr) {
	vti := link.(*Vti)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.IFLA_VTI_LOCAL:
			vti.Local = net.IP(datum.Value[0:4])
		case nl.IFLA_VTI_REMOTE:
			vti.Remote = net.IP(datum.Value[0:4])
		case nl.IFLA_VTI_IKEY:
			vti.IKey = ntohl(datum.Value[0:4])
		case nl.IFLA_VTI_OKEY:
			vti.OKey = ntohl(datum.Value[0:4])
		}
	}
}

func addVrfAttrs(vrf *Vrf, linkInfo *nl.RtAttr) {
	data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
	b := make([]byte, 4)
	native.PutUint32(b, uint32(vrf.Table))
	nl.NewRtAttrChild(data, nl.IFLA_VRF_TABLE, b)
}

func parseVrfData(link Link, data []syscall.NetlinkRouteAttr) {
	vrf := link.(*Vrf)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.IFLA_VRF_TABLE:
			vrf.Table = native.Uint32(datum.Value[0:4])
		}
	}
}

func addBridgeAttrs(bridge *Bridge, linkInfo *nl.RtAttr) {
	data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
	if bridge.MulticastSnooping != nil {
		nl.NewRtAttrChild(data, nl.IFLA_BR_MCAST_SNOOPING, boolToByte(*bridge.MulticastSnooping))
	}
	if bridge.HelloTime != nil {
		nl.NewRtAttrChild(data, nl.IFLA_BR_HELLO_TIME, nl.Uint32Attr(*bridge.HelloTime))
	}
}

func parseBridgeData(bridge Link, data []syscall.NetlinkRouteAttr) {
	br := bridge.(*Bridge)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.IFLA_BR_HELLO_TIME:
			helloTime := native.Uint32(datum.Value[0:4])
			br.HelloTime = &helloTime
		case nl.IFLA_BR_MCAST_SNOOPING:
			mcastSnooping := datum.Value[0] == 1
			br.MulticastSnooping = &mcastSnooping
		}
	}
}

func addGTPAttrs(gtp *GTP, linkInfo *nl.RtAttr) {
	data := nl.NewRtAttrChild(linkInfo, nl.IFLA_INFO_DATA, nil)
	nl.NewRtAttrChild(data, nl.IFLA_GTP_FD0, nl.Uint32Attr(uint32(gtp.FD0)))
	nl.NewRtAttrChild(data, nl.IFLA_GTP_FD1, nl.Uint32Attr(uint32(gtp.FD1)))
	nl.NewRtAttrChild(data, nl.IFLA_GTP_PDP_HASHSIZE, nl.Uint32Attr(131072))
	if gtp.Role != nl.GTP_ROLE_GGSN {
		nl.NewRtAttrChild(data, nl.IFLA_GTP_ROLE, nl.Uint32Attr(uint32(gtp.Role)))
	}
}

func parseGTPData(link Link, data []syscall.NetlinkRouteAttr) {
	gtp := link.(*GTP)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.IFLA_GTP_FD0:
			gtp.FD0 = int(native.Uint32(datum.Value))
		case nl.IFLA_GTP_FD1:
			gtp.FD1 = int(native.Uint32(datum.Value))
		case nl.IFLA_GTP_PDP_HASHSIZE:
			gtp.PDPHashsize = int(native.Uint32(datum.Value))
		case nl.IFLA_GTP_ROLE:
			gtp.Role = int(native.Uint32(datum.Value))
		}
	}
}
