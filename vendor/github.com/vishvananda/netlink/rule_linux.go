package netlink

import (
	"fmt"
	"net"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

const FibRuleInvert = 0x2

// RuleAdd adds a rule to the system.
// Equivalent to: ip rule add
func RuleAdd(rule *Rule) error {
	return pkgHandle.RuleAdd(rule)
}

// RuleAdd adds a rule to the system.
// Equivalent to: ip rule add
func (h *Handle) RuleAdd(rule *Rule) error {
	req := h.newNetlinkRequest(unix.RTM_NEWRULE, unix.NLM_F_CREATE|unix.NLM_F_EXCL|unix.NLM_F_ACK)
	return ruleHandle(rule, req)
}

// RuleDel deletes a rule from the system.
// Equivalent to: ip rule del
func RuleDel(rule *Rule) error {
	return pkgHandle.RuleDel(rule)
}

// RuleDel deletes a rule from the system.
// Equivalent to: ip rule del
func (h *Handle) RuleDel(rule *Rule) error {
	req := h.newNetlinkRequest(unix.RTM_DELRULE, unix.NLM_F_ACK)
	return ruleHandle(rule, req)
}

func ruleHandle(rule *Rule, req *nl.NetlinkRequest) error {
	msg := nl.NewRtMsg()
	msg.Family = unix.AF_INET
	msg.Protocol = unix.RTPROT_BOOT
	msg.Scope = unix.RT_SCOPE_UNIVERSE
	msg.Table = unix.RT_TABLE_UNSPEC
	msg.Type = unix.RTN_UNSPEC
	if req.NlMsghdr.Flags&unix.NLM_F_CREATE > 0 {
		msg.Type = unix.RTN_UNICAST
	}
	if rule.Invert {
		msg.Flags |= FibRuleInvert
	}
	if rule.Family != 0 {
		msg.Family = uint8(rule.Family)
	}
	if rule.Table >= 0 && rule.Table < 256 {
		msg.Table = uint8(rule.Table)
	}

	var dstFamily uint8
	var rtAttrs []*nl.RtAttr
	if rule.Dst != nil && rule.Dst.IP != nil {
		dstLen, _ := rule.Dst.Mask.Size()
		msg.Dst_len = uint8(dstLen)
		msg.Family = uint8(nl.GetIPFamily(rule.Dst.IP))
		dstFamily = msg.Family
		var dstData []byte
		if msg.Family == unix.AF_INET {
			dstData = rule.Dst.IP.To4()
		} else {
			dstData = rule.Dst.IP.To16()
		}
		rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_DST, dstData))
	}

	if rule.Src != nil && rule.Src.IP != nil {
		msg.Family = uint8(nl.GetIPFamily(rule.Src.IP))
		if dstFamily != 0 && dstFamily != msg.Family {
			return fmt.Errorf("source and destination ip are not the same IP family")
		}
		srcLen, _ := rule.Src.Mask.Size()
		msg.Src_len = uint8(srcLen)
		var srcData []byte
		if msg.Family == unix.AF_INET {
			srcData = rule.Src.IP.To4()
		} else {
			srcData = rule.Src.IP.To16()
		}
		rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_SRC, srcData))
	}

	req.AddData(msg)
	for i := range rtAttrs {
		req.AddData(rtAttrs[i])
	}

	native := nl.NativeEndian()

	if rule.Priority >= 0 {
		b := make([]byte, 4)
		native.PutUint32(b, uint32(rule.Priority))
		req.AddData(nl.NewRtAttr(nl.FRA_PRIORITY, b))
	}
	if rule.Mark >= 0 {
		b := make([]byte, 4)
		native.PutUint32(b, uint32(rule.Mark))
		req.AddData(nl.NewRtAttr(nl.FRA_FWMARK, b))
	}
	if rule.Mask >= 0 {
		b := make([]byte, 4)
		native.PutUint32(b, uint32(rule.Mask))
		req.AddData(nl.NewRtAttr(nl.FRA_FWMASK, b))
	}
	if rule.Flow >= 0 {
		b := make([]byte, 4)
		native.PutUint32(b, uint32(rule.Flow))
		req.AddData(nl.NewRtAttr(nl.FRA_FLOW, b))
	}
	if rule.TunID > 0 {
		b := make([]byte, 4)
		native.PutUint32(b, uint32(rule.TunID))
		req.AddData(nl.NewRtAttr(nl.FRA_TUN_ID, b))
	}
	if rule.Table >= 256 {
		b := make([]byte, 4)
		native.PutUint32(b, uint32(rule.Table))
		req.AddData(nl.NewRtAttr(nl.FRA_TABLE, b))
	}
	if msg.Table > 0 {
		if rule.SuppressPrefixlen >= 0 {
			b := make([]byte, 4)
			native.PutUint32(b, uint32(rule.SuppressPrefixlen))
			req.AddData(nl.NewRtAttr(nl.FRA_SUPPRESS_PREFIXLEN, b))
		}
		if rule.SuppressIfgroup >= 0 {
			b := make([]byte, 4)
			native.PutUint32(b, uint32(rule.SuppressIfgroup))
			req.AddData(nl.NewRtAttr(nl.FRA_SUPPRESS_IFGROUP, b))
		}
	}
	if rule.IifName != "" {
		req.AddData(nl.NewRtAttr(nl.FRA_IIFNAME, []byte(rule.IifName)))
	}
	if rule.OifName != "" {
		req.AddData(nl.NewRtAttr(nl.FRA_OIFNAME, []byte(rule.OifName)))
	}
	if rule.Goto >= 0 {
		msg.Type = nl.FR_ACT_GOTO
		b := make([]byte, 4)
		native.PutUint32(b, uint32(rule.Goto))
		req.AddData(nl.NewRtAttr(nl.FRA_GOTO, b))
	}

	_, err := req.Execute(unix.NETLINK_ROUTE, 0)
	return err
}

// RuleList lists rules in the system.
// Equivalent to: ip rule list
func RuleList(family int) ([]Rule, error) {
	return pkgHandle.RuleList(family)
}

// RuleList lists rules in the system.
// Equivalent to: ip rule list
func (h *Handle) RuleList(family int) ([]Rule, error) {
	req := h.newNetlinkRequest(unix.RTM_GETRULE, unix.NLM_F_DUMP|unix.NLM_F_REQUEST)
	msg := nl.NewIfInfomsg(family)
	req.AddData(msg)

	msgs, err := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWRULE)
	if err != nil {
		return nil, err
	}

	native := nl.NativeEndian()
	var res = make([]Rule, 0)
	for i := range msgs {
		msg := nl.DeserializeRtMsg(msgs[i])
		attrs, err := nl.ParseRouteAttr(msgs[i][msg.Len():])
		if err != nil {
			return nil, err
		}

		rule := NewRule()

		rule.Invert = msg.Flags&FibRuleInvert > 0

		for j := range attrs {
			switch attrs[j].Attr.Type {
			case unix.RTA_TABLE:
				rule.Table = int(native.Uint32(attrs[j].Value[0:4]))
			case nl.FRA_SRC:
				rule.Src = &net.IPNet{
					IP:   attrs[j].Value,
					Mask: net.CIDRMask(int(msg.Src_len), 8*len(attrs[j].Value)),
				}
			case nl.FRA_DST:
				rule.Dst = &net.IPNet{
					IP:   attrs[j].Value,
					Mask: net.CIDRMask(int(msg.Dst_len), 8*len(attrs[j].Value)),
				}
			case nl.FRA_FWMARK:
				rule.Mark = int(native.Uint32(attrs[j].Value[0:4]))
			case nl.FRA_FWMASK:
				rule.Mask = int(native.Uint32(attrs[j].Value[0:4]))
			case nl.FRA_TUN_ID:
				rule.TunID = uint(native.Uint64(attrs[j].Value[0:4]))
			case nl.FRA_IIFNAME:
				rule.IifName = string(attrs[j].Value[:len(attrs[j].Value)-1])
			case nl.FRA_OIFNAME:
				rule.OifName = string(attrs[j].Value[:len(attrs[j].Value)-1])
			case nl.FRA_SUPPRESS_PREFIXLEN:
				i := native.Uint32(attrs[j].Value[0:4])
				if i != 0xffffffff {
					rule.SuppressPrefixlen = int(i)
				}
			case nl.FRA_SUPPRESS_IFGROUP:
				i := native.Uint32(attrs[j].Value[0:4])
				if i != 0xffffffff {
					rule.SuppressIfgroup = int(i)
				}
			case nl.FRA_FLOW:
				rule.Flow = int(native.Uint32(attrs[j].Value[0:4]))
			case nl.FRA_GOTO:
				rule.Goto = int(native.Uint32(attrs[j].Value[0:4]))
			case nl.FRA_PRIORITY:
				rule.Priority = int(native.Uint32(attrs[j].Value[0:4]))
			}
		}
		res = append(res, *rule)
	}

	return res, nil
}
