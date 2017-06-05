package netlink

import (
	"fmt"
	"log"
	"net"
	"strings"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
)

// IFA_FLAGS is a u32 attribute.
const IFA_FLAGS = 0x8

// AddrAdd will add an IP address to a link device.
// Equivalent to: `ip addr add $addr dev $link`
func AddrAdd(link Link, addr *Addr) error {
	return pkgHandle.AddrAdd(link, addr)
}

// AddrAdd will add an IP address to a link device.
// Equivalent to: `ip addr add $addr dev $link`
func (h *Handle) AddrAdd(link Link, addr *Addr) error {
	req := h.newNetlinkRequest(syscall.RTM_NEWADDR, syscall.NLM_F_CREATE|syscall.NLM_F_EXCL|syscall.NLM_F_ACK)
	return h.addrHandle(link, addr, req)
}

// AddrReplace will replace (or, if not present, add) an IP address on a link device.
// Equivalent to: `ip addr replace $addr dev $link`
func AddrReplace(link Link, addr *Addr) error {
	return pkgHandle.AddrReplace(link, addr)
}

// AddrReplace will replace (or, if not present, add) an IP address on a link device.
// Equivalent to: `ip addr replace $addr dev $link`
func (h *Handle) AddrReplace(link Link, addr *Addr) error {
	req := h.newNetlinkRequest(syscall.RTM_NEWADDR, syscall.NLM_F_CREATE|syscall.NLM_F_REPLACE|syscall.NLM_F_ACK)
	return h.addrHandle(link, addr, req)
}

// AddrDel will delete an IP address from a link device.
// Equivalent to: `ip addr del $addr dev $link`
func AddrDel(link Link, addr *Addr) error {
	return pkgHandle.AddrDel(link, addr)
}

// AddrDel will delete an IP address from a link device.
// Equivalent to: `ip addr del $addr dev $link`
func (h *Handle) AddrDel(link Link, addr *Addr) error {
	req := h.newNetlinkRequest(syscall.RTM_DELADDR, syscall.NLM_F_ACK)
	return h.addrHandle(link, addr, req)
}

func (h *Handle) addrHandle(link Link, addr *Addr, req *nl.NetlinkRequest) error {
	base := link.Attrs()
	if addr.Label != "" && !strings.HasPrefix(addr.Label, base.Name) {
		return fmt.Errorf("label must begin with interface name")
	}
	h.ensureIndex(base)

	family := nl.GetIPFamily(addr.IP)

	msg := nl.NewIfAddrmsg(family)
	msg.Index = uint32(base.Index)
	msg.Scope = uint8(addr.Scope)
	prefixlen, _ := addr.Mask.Size()
	msg.Prefixlen = uint8(prefixlen)
	req.AddData(msg)

	var localAddrData []byte
	if family == FAMILY_V4 {
		localAddrData = addr.IP.To4()
	} else {
		localAddrData = addr.IP.To16()
	}

	localData := nl.NewRtAttr(syscall.IFA_LOCAL, localAddrData)
	req.AddData(localData)
	var peerAddrData []byte
	if addr.Peer != nil {
		if family == FAMILY_V4 {
			peerAddrData = addr.Peer.IP.To4()
		} else {
			peerAddrData = addr.Peer.IP.To16()
		}
	} else {
		peerAddrData = localAddrData
	}

	addressData := nl.NewRtAttr(syscall.IFA_ADDRESS, peerAddrData)
	req.AddData(addressData)

	if addr.Flags != 0 {
		if addr.Flags <= 0xff {
			msg.IfAddrmsg.Flags = uint8(addr.Flags)
		} else {
			b := make([]byte, 4)
			native.PutUint32(b, uint32(addr.Flags))
			flagsData := nl.NewRtAttr(IFA_FLAGS, b)
			req.AddData(flagsData)
		}
	}

	if addr.Broadcast != nil {
		req.AddData(nl.NewRtAttr(syscall.IFA_BROADCAST, addr.Broadcast))
	}

	if addr.Label != "" {
		labelData := nl.NewRtAttr(syscall.IFA_LABEL, nl.ZeroTerminated(addr.Label))
		req.AddData(labelData)
	}

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// AddrList gets a list of IP addresses in the system.
// Equivalent to: `ip addr show`.
// The list can be filtered by link and ip family.
func AddrList(link Link, family int) ([]Addr, error) {
	return pkgHandle.AddrList(link, family)
}

// AddrList gets a list of IP addresses in the system.
// Equivalent to: `ip addr show`.
// The list can be filtered by link and ip family.
func (h *Handle) AddrList(link Link, family int) ([]Addr, error) {
	req := h.newNetlinkRequest(syscall.RTM_GETADDR, syscall.NLM_F_DUMP)
	msg := nl.NewIfInfomsg(family)
	req.AddData(msg)

	msgs, err := req.Execute(syscall.NETLINK_ROUTE, syscall.RTM_NEWADDR)
	if err != nil {
		return nil, err
	}

	indexFilter := 0
	if link != nil {
		base := link.Attrs()
		h.ensureIndex(base)
		indexFilter = base.Index
	}

	var res []Addr
	for _, m := range msgs {
		addr, msgFamily, ifindex, err := parseAddr(m)
		if err != nil {
			return res, err
		}

		if link != nil && ifindex != indexFilter {
			// Ignore messages from other interfaces
			continue
		}

		if family != FAMILY_ALL && msgFamily != family {
			continue
		}

		res = append(res, addr)
	}

	return res, nil
}

func parseAddr(m []byte) (addr Addr, family, index int, err error) {
	msg := nl.DeserializeIfAddrmsg(m)

	family = -1
	index = -1

	attrs, err1 := nl.ParseRouteAttr(m[msg.Len():])
	if err1 != nil {
		err = err1
		return
	}

	family = int(msg.Family)
	index = int(msg.Index)

	var local, dst *net.IPNet
	for _, attr := range attrs {
		switch attr.Attr.Type {
		case syscall.IFA_ADDRESS:
			dst = &net.IPNet{
				IP:   attr.Value,
				Mask: net.CIDRMask(int(msg.Prefixlen), 8*len(attr.Value)),
			}
			addr.Peer = dst
		case syscall.IFA_LOCAL:
			local = &net.IPNet{
				IP:   attr.Value,
				Mask: net.CIDRMask(int(msg.Prefixlen), 8*len(attr.Value)),
			}
			addr.IPNet = local
		case syscall.IFA_BROADCAST:
			addr.Broadcast = attr.Value
		case syscall.IFA_LABEL:
			addr.Label = string(attr.Value[:len(attr.Value)-1])
		case IFA_FLAGS:
			addr.Flags = int(native.Uint32(attr.Value[0:4]))
		case nl.IFA_CACHEINFO:
			ci := nl.DeserializeIfaCacheInfo(attr.Value)
			addr.PreferedLft = int(ci.IfaPrefered)
			addr.ValidLft = int(ci.IfaValid)
		}
	}

	// IFA_LOCAL should be there but if not, fall back to IFA_ADDRESS
	if local != nil {
		addr.IPNet = local
	} else {
		addr.IPNet = dst
	}
	addr.Scope = int(msg.Scope)

	return
}

type AddrUpdate struct {
	LinkAddress net.IPNet
	LinkIndex   int
	Flags       int
	Scope       int
	PreferedLft int
	ValidLft    int
	NewAddr     bool // true=added false=deleted
}

// AddrSubscribe takes a chan down which notifications will be sent
// when addresses change.  Close the 'done' chan to stop subscription.
func AddrSubscribe(ch chan<- AddrUpdate, done <-chan struct{}) error {
	return addrSubscribe(netns.None(), netns.None(), ch, done)
}

// AddrSubscribeAt works like AddrSubscribe plus it allows the caller
// to choose the network namespace in which to subscribe (ns).
func AddrSubscribeAt(ns netns.NsHandle, ch chan<- AddrUpdate, done <-chan struct{}) error {
	return addrSubscribe(ns, netns.None(), ch, done)
}

func addrSubscribe(newNs, curNs netns.NsHandle, ch chan<- AddrUpdate, done <-chan struct{}) error {
	s, err := nl.SubscribeAt(newNs, curNs, syscall.NETLINK_ROUTE, syscall.RTNLGRP_IPV4_IFADDR, syscall.RTNLGRP_IPV6_IFADDR)
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
				log.Printf("netlink.AddrSubscribe: Receive() error: %v", err)
				return
			}
			for _, m := range msgs {
				msgType := m.Header.Type
				if msgType != syscall.RTM_NEWADDR && msgType != syscall.RTM_DELADDR {
					log.Printf("netlink.AddrSubscribe: bad message type: %d", msgType)
					continue
				}

				addr, _, ifindex, err := parseAddr(m.Data)
				if err != nil {
					log.Printf("netlink.AddrSubscribe: could not parse address: %v", err)
					continue
				}

				ch <- AddrUpdate{LinkAddress: *addr.IPNet,
					LinkIndex:   ifindex,
					NewAddr:     msgType == syscall.RTM_NEWADDR,
					Flags:       addr.Flags,
					Scope:       addr.Scope,
					PreferedLft: addr.PreferedLft,
					ValidLft:    addr.ValidLft}
			}
		}
	}()

	return nil
}
