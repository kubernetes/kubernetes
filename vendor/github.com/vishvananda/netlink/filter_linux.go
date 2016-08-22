package netlink

import (
	"fmt"
	"syscall"

	"github.com/vishvananda/netlink/nl"
)

// FilterDel will delete a filter from the system.
// Equivalent to: `tc filter del $filter`
func FilterDel(filter Filter) error {
	req := nl.NewNetlinkRequest(syscall.RTM_DELTFILTER, syscall.NLM_F_ACK)
	base := filter.Attrs()
	msg := &nl.TcMsg{
		Family:  nl.FAMILY_ALL,
		Ifindex: int32(base.LinkIndex),
		Handle:  base.Handle,
		Parent:  base.Parent,
		Info:    MakeHandle(base.Priority, nl.Swap16(base.Protocol)),
	}
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// FilterAdd will add a filter to the system.
// Equivalent to: `tc filter add $filter`
func FilterAdd(filter Filter) error {
	req := nl.NewNetlinkRequest(syscall.RTM_NEWTFILTER, syscall.NLM_F_CREATE|syscall.NLM_F_EXCL|syscall.NLM_F_ACK)
	base := filter.Attrs()
	msg := &nl.TcMsg{
		Family:  nl.FAMILY_ALL,
		Ifindex: int32(base.LinkIndex),
		Handle:  base.Handle,
		Parent:  base.Parent,
		Info:    MakeHandle(base.Priority, nl.Swap16(base.Protocol)),
	}
	req.AddData(msg)
	req.AddData(nl.NewRtAttr(nl.TCA_KIND, nl.ZeroTerminated(filter.Type())))

	options := nl.NewRtAttr(nl.TCA_OPTIONS, nil)
	if u32, ok := filter.(*U32); ok {
		// match all
		sel := nl.TcU32Sel{
			Nkeys: 1,
			Flags: nl.TC_U32_TERMINAL,
		}
		sel.Keys = append(sel.Keys, nl.TcU32Key{})
		nl.NewRtAttrChild(options, nl.TCA_U32_SEL, sel.Serialize())
		actions := nl.NewRtAttrChild(options, nl.TCA_U32_ACT, nil)
		table := nl.NewRtAttrChild(actions, nl.TCA_ACT_TAB, nil)
		nl.NewRtAttrChild(table, nl.TCA_KIND, nl.ZeroTerminated("mirred"))
		// redirect to other interface
		mir := nl.TcMirred{
			Action:  nl.TC_ACT_STOLEN,
			Eaction: nl.TCA_EGRESS_REDIR,
			Ifindex: uint32(u32.RedirIndex),
		}
		aopts := nl.NewRtAttrChild(table, nl.TCA_OPTIONS, nil)
		nl.NewRtAttrChild(aopts, nl.TCA_MIRRED_PARMS, mir.Serialize())
	}
	req.AddData(options)
	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// FilterList gets a list of filters in the system.
// Equivalent to: `tc filter show`.
// Generally retunrs nothing if link and parent are not specified.
func FilterList(link Link, parent uint32) ([]Filter, error) {
	req := nl.NewNetlinkRequest(syscall.RTM_GETTFILTER, syscall.NLM_F_DUMP)
	msg := &nl.TcMsg{
		Family: nl.FAMILY_ALL,
		Parent: parent,
	}
	if link != nil {
		base := link.Attrs()
		ensureIndex(base)
		msg.Ifindex = int32(base.Index)
	}
	req.AddData(msg)

	msgs, err := req.Execute(syscall.NETLINK_ROUTE, syscall.RTM_NEWTFILTER)
	if err != nil {
		return nil, err
	}

	var res []Filter
	for _, m := range msgs {
		msg := nl.DeserializeTcMsg(m)

		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return nil, err
		}

		base := FilterAttrs{
			LinkIndex: int(msg.Ifindex),
			Handle:    msg.Handle,
			Parent:    msg.Parent,
		}
		base.Priority, base.Protocol = MajorMinor(msg.Info)
		base.Protocol = nl.Swap16(base.Protocol)

		var filter Filter
		filterType := ""
		detailed := false
		for _, attr := range attrs {
			switch attr.Attr.Type {
			case nl.TCA_KIND:
				filterType = string(attr.Value[:len(attr.Value)-1])
				switch filterType {
				case "u32":
					filter = &U32{}
				default:
					filter = &GenericFilter{FilterType: filterType}
				}
			case nl.TCA_OPTIONS:
				switch filterType {
				case "u32":
					data, err := nl.ParseRouteAttr(attr.Value)
					if err != nil {
						return nil, err
					}
					detailed, err = parseU32Data(filter, data)
					if err != nil {
						return nil, err
					}
				}
			}
		}
		// only return the detailed version of the filter
		if detailed {
			*filter.Attrs() = base
			res = append(res, filter)
		}
	}

	return res, nil
}

func parseU32Data(filter Filter, data []syscall.NetlinkRouteAttr) (bool, error) {
	native = nl.NativeEndian()
	u32 := filter.(*U32)
	detailed := false
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_U32_SEL:
			detailed = true
			sel := nl.DeserializeTcU32Sel(datum.Value)
			// only parse if we have a very basic redirect
			if sel.Flags&nl.TC_U32_TERMINAL == 0 || sel.Nkeys != 1 {
				return detailed, nil
			}
		case nl.TCA_U32_ACT:
			table, err := nl.ParseRouteAttr(datum.Value)
			if err != nil {
				return detailed, err
			}
			if len(table) != 1 || table[0].Attr.Type != nl.TCA_ACT_TAB {
				return detailed, fmt.Errorf("Action table not formed properly")
			}
			aattrs, err := nl.ParseRouteAttr(table[0].Value)
			for _, aattr := range aattrs {
				switch aattr.Attr.Type {
				case nl.TCA_KIND:
					actionType := string(aattr.Value[:len(aattr.Value)-1])
					// only parse if the action is mirred
					if actionType != "mirred" {
						return detailed, nil
					}
				case nl.TCA_OPTIONS:
					adata, err := nl.ParseRouteAttr(aattr.Value)
					if err != nil {
						return detailed, err
					}
					for _, adatum := range adata {
						switch adatum.Attr.Type {
						case nl.TCA_MIRRED_PARMS:
							mir := nl.DeserializeTcMirred(adatum.Value)
							u32.RedirIndex = int(mir.Ifindex)
						}
					}
				}
			}
		}
	}
	return detailed, nil
}
