package netlink

import (
	"fmt"
	"net"
	"strings"
	"syscall"

	"github.com/vishvananda/netlink/nl"
)

// AddrAdd will add an IP address to a link device.
// Equivalent to: `ip addr add $addr dev $link`
func AddrAdd(link Link, addr *Addr) error {

	req := nl.NewNetlinkRequest(syscall.RTM_NEWADDR, syscall.NLM_F_CREATE|syscall.NLM_F_EXCL|syscall.NLM_F_ACK)
	return addrHandle(link, addr, req)
}

// AddrDel will delete an IP address from a link device.
// Equivalent to: `ip addr del $addr dev $link`
func AddrDel(link Link, addr *Addr) error {
	req := nl.NewNetlinkRequest(syscall.RTM_DELADDR, syscall.NLM_F_ACK)
	return addrHandle(link, addr, req)
}

func addrHandle(link Link, addr *Addr, req *nl.NetlinkRequest) error {
	base := link.Attrs()
	if addr.Label != "" && !strings.HasPrefix(addr.Label, base.Name) {
		return fmt.Errorf("label must begin with interface name")
	}
	ensureIndex(base)

	family := nl.GetIPFamily(addr.IP)

	msg := nl.NewIfAddrmsg(family)
	msg.Index = uint32(base.Index)
	prefixlen, _ := addr.Mask.Size()
	msg.Prefixlen = uint8(prefixlen)
	req.AddData(msg)

	var addrData []byte
	if family == FAMILY_V4 {
		addrData = addr.IP.To4()
	} else {
		addrData = addr.IP.To16()
	}

	localData := nl.NewRtAttr(syscall.IFA_LOCAL, addrData)
	req.AddData(localData)

	addressData := nl.NewRtAttr(syscall.IFA_ADDRESS, addrData)
	req.AddData(addressData)

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
	req := nl.NewNetlinkRequest(syscall.RTM_GETADDR, syscall.NLM_F_DUMP)
	msg := nl.NewIfInfomsg(family)
	req.AddData(msg)

	msgs, err := req.Execute(syscall.NETLINK_ROUTE, syscall.RTM_NEWADDR)
	if err != nil {
		return nil, err
	}

	index := 0
	if link != nil {
		base := link.Attrs()
		ensureIndex(base)
		index = base.Index
	}

	var res []Addr
	for _, m := range msgs {
		msg := nl.DeserializeIfAddrmsg(m)

		if link != nil && msg.Index != uint32(index) {
			// Ignore messages from other interfaces
			continue
		}

		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return nil, err
		}

		var local, dst *net.IPNet
		var addr Addr
		for _, attr := range attrs {
			switch attr.Attr.Type {
			case syscall.IFA_ADDRESS:
				dst = &net.IPNet{
					IP:   attr.Value,
					Mask: net.CIDRMask(int(msg.Prefixlen), 8*len(attr.Value)),
				}
			case syscall.IFA_LOCAL:
				local = &net.IPNet{
					IP:   attr.Value,
					Mask: net.CIDRMask(int(msg.Prefixlen), 8*len(attr.Value)),
				}
			case syscall.IFA_LABEL:
				addr.Label = string(attr.Value[:len(attr.Value)-1])
			}
		}

		// IFA_LOCAL should be there but if not, fall back to IFA_ADDRESS
		if local != nil {
			addr.IPNet = local
		} else {
			addr.IPNet = dst
		}

		res = append(res, addr)
	}

	return res, nil
}
