package netlink

import (
	"syscall"

	"github.com/vishvananda/netlink/nl"
)

func selFromPolicy(sel *nl.XfrmSelector, policy *XfrmPolicy) {
	sel.Family = uint16(nl.GetIPFamily(policy.Dst.IP))
	sel.Daddr.FromIP(policy.Dst.IP)
	sel.Saddr.FromIP(policy.Src.IP)
	prefixlenD, _ := policy.Dst.Mask.Size()
	sel.PrefixlenD = uint8(prefixlenD)
	prefixlenS, _ := policy.Src.Mask.Size()
	sel.PrefixlenS = uint8(prefixlenS)
}

// XfrmPolicyAdd will add an xfrm policy to the system.
// Equivalent to: `ip xfrm policy add $policy`
func XfrmPolicyAdd(policy *XfrmPolicy) error {
	req := nl.NewNetlinkRequest(nl.XFRM_MSG_NEWPOLICY, syscall.NLM_F_CREATE|syscall.NLM_F_EXCL|syscall.NLM_F_ACK)

	msg := &nl.XfrmUserpolicyInfo{}
	selFromPolicy(&msg.Sel, policy)
	msg.Priority = uint32(policy.Priority)
	msg.Index = uint32(policy.Index)
	msg.Dir = uint8(policy.Dir)
	msg.Lft.SoftByteLimit = nl.XFRM_INF
	msg.Lft.HardByteLimit = nl.XFRM_INF
	msg.Lft.SoftPacketLimit = nl.XFRM_INF
	msg.Lft.HardPacketLimit = nl.XFRM_INF
	req.AddData(msg)

	tmplData := make([]byte, nl.SizeofXfrmUserTmpl*len(policy.Tmpls))
	for i, tmpl := range policy.Tmpls {
		start := i * nl.SizeofXfrmUserTmpl
		userTmpl := nl.DeserializeXfrmUserTmpl(tmplData[start : start+nl.SizeofXfrmUserTmpl])
		userTmpl.XfrmId.Daddr.FromIP(tmpl.Dst)
		userTmpl.Saddr.FromIP(tmpl.Src)
		userTmpl.XfrmId.Proto = uint8(tmpl.Proto)
		userTmpl.Mode = uint8(tmpl.Mode)
		userTmpl.Reqid = uint32(tmpl.Reqid)
		userTmpl.Aalgos = ^uint32(0)
		userTmpl.Ealgos = ^uint32(0)
		userTmpl.Calgos = ^uint32(0)
	}
	if len(tmplData) > 0 {
		tmpls := nl.NewRtAttr(nl.XFRMA_TMPL, tmplData)
		req.AddData(tmpls)
	}

	_, err := req.Execute(syscall.NETLINK_XFRM, 0)
	return err
}

// XfrmPolicyDel will delete an xfrm policy from the system. Note that
// the Tmpls are ignored when matching the policy to delete.
// Equivalent to: `ip xfrm policy del $policy`
func XfrmPolicyDel(policy *XfrmPolicy) error {
	req := nl.NewNetlinkRequest(nl.XFRM_MSG_DELPOLICY, syscall.NLM_F_ACK)

	msg := &nl.XfrmUserpolicyId{}
	selFromPolicy(&msg.Sel, policy)
	msg.Index = uint32(policy.Index)
	msg.Dir = uint8(policy.Dir)
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_XFRM, 0)
	return err
}

// XfrmPolicyList gets a list of xfrm policies in the system.
// Equivalent to: `ip xfrm policy show`.
// The list can be filtered by ip family.
func XfrmPolicyList(family int) ([]XfrmPolicy, error) {
	req := nl.NewNetlinkRequest(nl.XFRM_MSG_GETPOLICY, syscall.NLM_F_DUMP)

	msg := nl.NewIfInfomsg(family)
	req.AddData(msg)

	msgs, err := req.Execute(syscall.NETLINK_XFRM, nl.XFRM_MSG_NEWPOLICY)
	if err != nil {
		return nil, err
	}

	var res []XfrmPolicy
	for _, m := range msgs {
		msg := nl.DeserializeXfrmUserpolicyInfo(m)

		if family != FAMILY_ALL && family != int(msg.Sel.Family) {
			continue
		}

		var policy XfrmPolicy

		policy.Dst = msg.Sel.Daddr.ToIPNet(msg.Sel.PrefixlenD)
		policy.Src = msg.Sel.Saddr.ToIPNet(msg.Sel.PrefixlenS)
		policy.Priority = int(msg.Priority)
		policy.Index = int(msg.Index)
		policy.Dir = Dir(msg.Dir)

		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return nil, err
		}

		for _, attr := range attrs {
			switch attr.Attr.Type {
			case nl.XFRMA_TMPL:
				max := len(attr.Value)
				for i := 0; i < max; i += nl.SizeofXfrmUserTmpl {
					var resTmpl XfrmPolicyTmpl
					tmpl := nl.DeserializeXfrmUserTmpl(attr.Value[i : i+nl.SizeofXfrmUserTmpl])
					resTmpl.Dst = tmpl.XfrmId.Daddr.ToIP()
					resTmpl.Src = tmpl.Saddr.ToIP()
					resTmpl.Proto = Proto(tmpl.XfrmId.Proto)
					resTmpl.Mode = Mode(tmpl.Mode)
					resTmpl.Reqid = int(tmpl.Reqid)
					policy.Tmpls = append(policy.Tmpls, resTmpl)
				}
			}
		}
		res = append(res, policy)
	}
	return res, nil
}
