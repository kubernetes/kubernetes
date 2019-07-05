package netlink

import (
	"net"
	"syscall"
	"unsafe"

	"github.com/vishvananda/netlink/nl"
)

const (
	NDA_UNSPEC = iota
	NDA_DST
	NDA_LLADDR
	NDA_CACHEINFO
	NDA_PROBES
	NDA_VLAN
	NDA_PORT
	NDA_VNI
	NDA_IFINDEX
	NDA_MAX = NDA_IFINDEX
)

// Neighbor Cache Entry States.
const (
	NUD_NONE       = 0x00
	NUD_INCOMPLETE = 0x01
	NUD_REACHABLE  = 0x02
	NUD_STALE      = 0x04
	NUD_DELAY      = 0x08
	NUD_PROBE      = 0x10
	NUD_FAILED     = 0x20
	NUD_NOARP      = 0x40
	NUD_PERMANENT  = 0x80
)

// Neighbor Flags
const (
	NTF_USE    = 0x01
	NTF_SELF   = 0x02
	NTF_MASTER = 0x04
	NTF_PROXY  = 0x08
	NTF_ROUTER = 0x80
)

type Ndmsg struct {
	Family uint8
	Index  uint32
	State  uint16
	Flags  uint8
	Type   uint8
}

func deserializeNdmsg(b []byte) *Ndmsg {
	var dummy Ndmsg
	return (*Ndmsg)(unsafe.Pointer(&b[0:unsafe.Sizeof(dummy)][0]))
}

func (msg *Ndmsg) Serialize() []byte {
	return (*(*[unsafe.Sizeof(*msg)]byte)(unsafe.Pointer(msg)))[:]
}

func (msg *Ndmsg) Len() int {
	return int(unsafe.Sizeof(*msg))
}

// NeighAdd will add an IP to MAC mapping to the ARP table
// Equivalent to: `ip neigh add ....`
func NeighAdd(neigh *Neigh) error {
	return pkgHandle.NeighAdd(neigh)
}

// NeighAdd will add an IP to MAC mapping to the ARP table
// Equivalent to: `ip neigh add ....`
func (h *Handle) NeighAdd(neigh *Neigh) error {
	return h.neighAdd(neigh, syscall.NLM_F_CREATE|syscall.NLM_F_EXCL)
}

// NeighSet will add or replace an IP to MAC mapping to the ARP table
// Equivalent to: `ip neigh replace....`
func NeighSet(neigh *Neigh) error {
	return pkgHandle.NeighSet(neigh)
}

// NeighSet will add or replace an IP to MAC mapping to the ARP table
// Equivalent to: `ip neigh replace....`
func (h *Handle) NeighSet(neigh *Neigh) error {
	return h.neighAdd(neigh, syscall.NLM_F_CREATE|syscall.NLM_F_REPLACE)
}

// NeighAppend will append an entry to FDB
// Equivalent to: `bridge fdb append...`
func NeighAppend(neigh *Neigh) error {
	return pkgHandle.NeighAppend(neigh)
}

// NeighAppend will append an entry to FDB
// Equivalent to: `bridge fdb append...`
func (h *Handle) NeighAppend(neigh *Neigh) error {
	return h.neighAdd(neigh, syscall.NLM_F_CREATE|syscall.NLM_F_APPEND)
}

// NeighAppend will append an entry to FDB
// Equivalent to: `bridge fdb append...`
func neighAdd(neigh *Neigh, mode int) error {
	return pkgHandle.neighAdd(neigh, mode)
}

// NeighAppend will append an entry to FDB
// Equivalent to: `bridge fdb append...`
func (h *Handle) neighAdd(neigh *Neigh, mode int) error {
	req := h.newNetlinkRequest(syscall.RTM_NEWNEIGH, mode|syscall.NLM_F_ACK)
	return neighHandle(neigh, req)
}

// NeighDel will delete an IP address from a link device.
// Equivalent to: `ip addr del $addr dev $link`
func NeighDel(neigh *Neigh) error {
	return pkgHandle.NeighDel(neigh)
}

// NeighDel will delete an IP address from a link device.
// Equivalent to: `ip addr del $addr dev $link`
func (h *Handle) NeighDel(neigh *Neigh) error {
	req := h.newNetlinkRequest(syscall.RTM_DELNEIGH, syscall.NLM_F_ACK)
	return neighHandle(neigh, req)
}

func neighHandle(neigh *Neigh, req *nl.NetlinkRequest) error {
	var family int

	if neigh.Family > 0 {
		family = neigh.Family
	} else {
		family = nl.GetIPFamily(neigh.IP)
	}

	msg := Ndmsg{
		Family: uint8(family),
		Index:  uint32(neigh.LinkIndex),
		State:  uint16(neigh.State),
		Type:   uint8(neigh.Type),
		Flags:  uint8(neigh.Flags),
	}
	req.AddData(&msg)

	ipData := neigh.IP.To4()
	if ipData == nil {
		ipData = neigh.IP.To16()
	}

	dstData := nl.NewRtAttr(NDA_DST, ipData)
	req.AddData(dstData)

	if neigh.LLIPAddr != nil {
		llIPData := nl.NewRtAttr(NDA_LLADDR, neigh.LLIPAddr.To4())
		req.AddData(llIPData)
	} else if neigh.Flags != NTF_PROXY || neigh.HardwareAddr != nil {
		hwData := nl.NewRtAttr(NDA_LLADDR, []byte(neigh.HardwareAddr))
		req.AddData(hwData)
	}

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// NeighList gets a list of IP-MAC mappings in the system (ARP table).
// Equivalent to: `ip neighbor show`.
// The list can be filtered by link and ip family.
func NeighList(linkIndex, family int) ([]Neigh, error) {
	return pkgHandle.NeighList(linkIndex, family)
}

// NeighProxyList gets a list of neighbor proxies in the system.
// Equivalent to: `ip neighbor show proxy`.
// The list can be filtered by link and ip family.
func NeighProxyList(linkIndex, family int) ([]Neigh, error) {
	return pkgHandle.NeighProxyList(linkIndex, family)
}

// NeighList gets a list of IP-MAC mappings in the system (ARP table).
// Equivalent to: `ip neighbor show`.
// The list can be filtered by link and ip family.
func (h *Handle) NeighList(linkIndex, family int) ([]Neigh, error) {
	return h.neighList(linkIndex, family, 0)
}

// NeighProxyList gets a list of neighbor proxies in the system.
// Equivalent to: `ip neighbor show proxy`.
// The list can be filtered by link, ip family.
func (h *Handle) NeighProxyList(linkIndex, family int) ([]Neigh, error) {
	return h.neighList(linkIndex, family, NTF_PROXY)
}

func (h *Handle) neighList(linkIndex, family, flags int) ([]Neigh, error) {
	req := h.newNetlinkRequest(syscall.RTM_GETNEIGH, syscall.NLM_F_DUMP)
	msg := Ndmsg{
		Family: uint8(family),
		Index:  uint32(linkIndex),
		Flags:  uint8(flags),
	}
	req.AddData(&msg)

	msgs, err := req.Execute(syscall.NETLINK_ROUTE, syscall.RTM_NEWNEIGH)
	if err != nil {
		return nil, err
	}

	var res []Neigh
	for _, m := range msgs {
		ndm := deserializeNdmsg(m)
		if linkIndex != 0 && int(ndm.Index) != linkIndex {
			// Ignore messages from other interfaces
			continue
		}

		neigh, err := NeighDeserialize(m)
		if err != nil {
			continue
		}

		res = append(res, *neigh)
	}

	return res, nil
}

func NeighDeserialize(m []byte) (*Neigh, error) {
	msg := deserializeNdmsg(m)

	neigh := Neigh{
		LinkIndex: int(msg.Index),
		Family:    int(msg.Family),
		State:     int(msg.State),
		Type:      int(msg.Type),
		Flags:     int(msg.Flags),
	}

	attrs, err := nl.ParseRouteAttr(m[msg.Len():])
	if err != nil {
		return nil, err
	}

	// This should be cached for perfomance
	// once per table dump
	link, err := LinkByIndex(neigh.LinkIndex)
	if err != nil {
		return nil, err
	}
	encapType := link.Attrs().EncapType

	for _, attr := range attrs {
		switch attr.Attr.Type {
		case NDA_DST:
			neigh.IP = net.IP(attr.Value)
		case NDA_LLADDR:
			// BUG: Is this a bug in the netlink library?
			// #define RTA_LENGTH(len) (RTA_ALIGN(sizeof(struct rtattr)) + (len))
			// #define RTA_PAYLOAD(rta) ((int)((rta)->rta_len) - RTA_LENGTH(0))
			attrLen := attr.Attr.Len - syscall.SizeofRtAttr
			if attrLen == 4 && (encapType == "ipip" ||
				encapType == "sit" ||
				encapType == "gre") {
				neigh.LLIPAddr = net.IP(attr.Value)
			} else if attrLen == 16 &&
				encapType == "tunnel6" {
				neigh.IP = net.IP(attr.Value)
			} else {
				neigh.HardwareAddr = net.HardwareAddr(attr.Value)
			}
		}
	}

	return &neigh, nil
}
