package netlink

import (
	"errors"
	"fmt"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

// BridgeVlanTunnelShow gets vlanid-tunnelid mapping.
// Equivalent to: `bridge vlan tunnelshow`
//
// If the returned error is [ErrDumpInterrupted], results may be inconsistent
// or incomplete.
func BridgeVlanTunnelShow() ([]nl.TunnelInfo, error) {
	return pkgHandle.BridgeVlanTunnelShow()
}

func (h *Handle) BridgeVlanTunnelShow() ([]nl.TunnelInfo, error) {
	req := h.newNetlinkRequest(unix.RTM_GETLINK, unix.NLM_F_DUMP)
	msg := nl.NewIfInfomsg(unix.AF_BRIDGE)
	req.AddData(msg)
	req.AddData(nl.NewRtAttr(unix.IFLA_EXT_MASK, nl.Uint32Attr(uint32(nl.RTEXT_FILTER_BRVLAN))))

	msgs, executeErr := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWLINK)
	if executeErr != nil && !errors.Is(executeErr, ErrDumpInterrupted) {
		return nil, executeErr
	}
	ret := make([]nl.TunnelInfo, 0)
	for _, m := range msgs {
		msg := nl.DeserializeIfInfomsg(m)

		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return nil, err
		}
		for _, attr := range attrs {
			switch attr.Attr.Type {
			case unix.IFLA_AF_SPEC:
				nestedAttrs, err := nl.ParseRouteAttr(attr.Value)
				if err != nil {
					return nil, fmt.Errorf("failed to parse nested attr %v", err)
				}
				for _, nestAttr := range nestedAttrs {
					switch nestAttr.Attr.Type {
					case nl.IFLA_BRIDGE_VLAN_TUNNEL_INFO:
						ret, err = parseTunnelInfo(&nestAttr, ret)
						if err != nil {
							return nil, fmt.Errorf("failed to parse tunnelinfo %v", err)
						}
					}
				}
			}
		}
	}
	return ret, executeErr
}

func parseTunnelInfo(nestAttr *syscall.NetlinkRouteAttr, results []nl.TunnelInfo) ([]nl.TunnelInfo, error) {
	tunnelInfos, err := nl.ParseRouteAttr(nestAttr.Value)
	if err != nil {
		return nil, fmt.Errorf("failed to parse nested attr %v", err)
	}
	var tunnelId uint32
	var vid uint16
	var flag uint16
	for _, tunnelInfo := range tunnelInfos {
		switch tunnelInfo.Attr.Type {
		case nl.IFLA_BRIDGE_VLAN_TUNNEL_ID:
			tunnelId = native.Uint32(tunnelInfo.Value)
		case nl.IFLA_BRIDGE_VLAN_TUNNEL_VID:
			vid = native.Uint16(tunnelInfo.Value)
		case nl.IFLA_BRIDGE_VLAN_TUNNEL_FLAGS:
			flag = native.Uint16(tunnelInfo.Value)
		}
	}

	if flag == nl.BRIDGE_VLAN_INFO_RANGE_END {
		lastTi := results[len(results)-1]
		vni := lastTi.TunId + 1
		for i := lastTi.Vid + 1; i < vid; i++ {
			t := nl.TunnelInfo{
				TunId: vni,
				Vid:   i,
			}
			results = append(results, t)
			vni++
		}
	}

	t := nl.TunnelInfo{
		TunId: tunnelId,
		Vid:   vid,
	}

	results = append(results, t)
	return results, nil
}

// BridgeVlanList gets a map of device id to bridge vlan infos.
// Equivalent to: `bridge vlan show`
//
// If the returned error is [ErrDumpInterrupted], results may be inconsistent
// or incomplete.
func BridgeVlanList() (map[int32][]*nl.BridgeVlanInfo, error) {
	return pkgHandle.BridgeVlanList()
}

// BridgeVlanList gets a map of device id to bridge vlan infos.
// Equivalent to: `bridge vlan show`
//
// If the returned error is [ErrDumpInterrupted], results may be inconsistent
// or incomplete.
func (h *Handle) BridgeVlanList() (map[int32][]*nl.BridgeVlanInfo, error) {
	req := h.newNetlinkRequest(unix.RTM_GETLINK, unix.NLM_F_DUMP)
	msg := nl.NewIfInfomsg(unix.AF_BRIDGE)
	req.AddData(msg)
	req.AddData(nl.NewRtAttr(unix.IFLA_EXT_MASK, nl.Uint32Attr(uint32(nl.RTEXT_FILTER_BRVLAN))))

	msgs, executeErr := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWLINK)
	if executeErr != nil && !errors.Is(executeErr, ErrDumpInterrupted) {
		return nil, executeErr
	}
	ret := make(map[int32][]*nl.BridgeVlanInfo)
	for _, m := range msgs {
		msg := nl.DeserializeIfInfomsg(m)

		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return nil, err
		}
		for _, attr := range attrs {
			switch attr.Attr.Type {
			case unix.IFLA_AF_SPEC:
				//nested attr
				nestAttrs, err := nl.ParseRouteAttr(attr.Value)
				if err != nil {
					return nil, fmt.Errorf("failed to parse nested attr %v", err)
				}
				for _, nestAttr := range nestAttrs {
					switch nestAttr.Attr.Type {
					case nl.IFLA_BRIDGE_VLAN_INFO:
						vlanInfo := nl.DeserializeBridgeVlanInfo(nestAttr.Value)
						ret[msg.Index] = append(ret[msg.Index], vlanInfo)
					}
				}
			}
		}
	}
	return ret, executeErr
}

// BridgeVlanAddTunnelInfo adds a new vlan filter entry
// Equivalent to: `bridge vlan add dev DEV vid VID tunnel_info id TUNID [ self ] [ master ]`
func BridgeVlanAddTunnelInfo(link Link, vid uint16, tunid uint32, self, master bool) error {
	return pkgHandle.BridgeVlanAddTunnelInfo(link, vid, 0, tunid, 0, self, master)
}

// BridgeVlanAddRangeTunnelInfoRange adds a new vlan filter entry
// Equivalent to: `bridge vlan add dev DEV vid VID-VIDEND tunnel_info id VIN-VINEND [ self ] [ master ]`
func BridgeVlanAddRangeTunnelInfoRange(link Link, vid, vidEnd uint16, tunid, tunidEnd uint32, self, master bool) error {
	return pkgHandle.BridgeVlanAddTunnelInfo(link, vid, vidEnd, tunid, tunidEnd, self, master)
}

func (h *Handle) BridgeVlanAddTunnelInfo(link Link, vid, vidEnd uint16, tunid, tunidEnd uint32, self, master bool) error {
	return h.bridgeVlanModify(unix.RTM_SETLINK, link, vid, vidEnd, tunid, tunidEnd, false, false, self, master)
}

// BridgeVlanDelTunnelInfo adds a new vlan filter entry
// Equivalent to: `bridge vlan del dev DEV vid VID tunnel_info id TUNID [ self ] [ master ]`
func BridgeVlanDelTunnelInfo(link Link, vid uint16, tunid uint32, self, master bool) error {
	return pkgHandle.BridgeVlanDelTunnelInfo(link, vid, 0, tunid, 0, self, master)
}

// BridgeVlanDelRangeTunnelInfoRange adds a new vlan filter entry
// Equivalent to: `bridge vlan del dev DEV vid VID-VIDEND tunnel_info id VIN-VINEND [ self ] [ master ]`
func BridgeVlanDelRangeTunnelInfoRange(link Link, vid, vidEnd uint16, tunid, tunidEnd uint32, self, master bool) error {
	return pkgHandle.BridgeVlanDelTunnelInfo(link, vid, vidEnd, tunid, tunidEnd, self, master)
}

func (h *Handle) BridgeVlanDelTunnelInfo(link Link, vid, vidEnd uint16, tunid, tunidEnd uint32, self, master bool) error {
	return h.bridgeVlanModify(unix.RTM_DELLINK, link, vid, vidEnd, tunid, tunidEnd, false, false, self, master)
}

// BridgeVlanAdd adds a new vlan filter entry
// Equivalent to: `bridge vlan add dev DEV vid VID [ pvid ] [ untagged ] [ self ] [ master ]`
func BridgeVlanAdd(link Link, vid uint16, pvid, untagged, self, master bool) error {
	return pkgHandle.BridgeVlanAdd(link, vid, pvid, untagged, self, master)
}

// BridgeVlanAdd adds a new vlan filter entry
// Equivalent to: `bridge vlan add dev DEV vid VID [ pvid ] [ untagged ] [ self ] [ master ]`
func (h *Handle) BridgeVlanAdd(link Link, vid uint16, pvid, untagged, self, master bool) error {
	return h.bridgeVlanModify(unix.RTM_SETLINK, link, vid, 0, 0, 0, pvid, untagged, self, master)
}

// BridgeVlanAddRange adds a new vlan filter entry
// Equivalent to: `bridge vlan add dev DEV vid VID-VIDEND [ pvid ] [ untagged ] [ self ] [ master ]`
func BridgeVlanAddRange(link Link, vid, vidEnd uint16, pvid, untagged, self, master bool) error {
	return pkgHandle.BridgeVlanAddRange(link, vid, vidEnd, pvid, untagged, self, master)
}

// BridgeVlanAddRange adds a new vlan filter entry
// Equivalent to: `bridge vlan add dev DEV vid VID-VIDEND [ pvid ] [ untagged ] [ self ] [ master ]`
func (h *Handle) BridgeVlanAddRange(link Link, vid, vidEnd uint16, pvid, untagged, self, master bool) error {
	return h.bridgeVlanModify(unix.RTM_SETLINK, link, vid, vidEnd, 0, 0, pvid, untagged, self, master)
}

// BridgeVlanDel adds a new vlan filter entry
// Equivalent to: `bridge vlan del dev DEV vid VID [ pvid ] [ untagged ] [ self ] [ master ]`
func BridgeVlanDel(link Link, vid uint16, pvid, untagged, self, master bool) error {
	return pkgHandle.BridgeVlanDel(link, vid, pvid, untagged, self, master)
}

// BridgeVlanDel adds a new vlan filter entry
// Equivalent to: `bridge vlan del dev DEV vid VID [ pvid ] [ untagged ] [ self ] [ master ]`
func (h *Handle) BridgeVlanDel(link Link, vid uint16, pvid, untagged, self, master bool) error {
	return h.bridgeVlanModify(unix.RTM_DELLINK, link, vid, 0, 0, 0, pvid, untagged, self, master)
}

// BridgeVlanDelRange adds a new vlan filter entry
// Equivalent to: `bridge vlan del dev DEV vid VID-VIDEND [ pvid ] [ untagged ] [ self ] [ master ]`
func BridgeVlanDelRange(link Link, vid, vidEnd uint16, pvid, untagged, self, master bool) error {
	return pkgHandle.BridgeVlanDelRange(link, vid, vidEnd, pvid, untagged, self, master)
}

// BridgeVlanDelRange adds a new vlan filter entry
// Equivalent to: `bridge vlan del dev DEV vid VID-VIDEND [ pvid ] [ untagged ] [ self ] [ master ]`
func (h *Handle) BridgeVlanDelRange(link Link, vid, vidEnd uint16, pvid, untagged, self, master bool) error {
	return h.bridgeVlanModify(unix.RTM_DELLINK, link, vid, vidEnd, 0, 0, pvid, untagged, self, master)
}

func (h *Handle) bridgeVlanModify(cmd int, link Link, vid, vidEnd uint16, tunid, tunidEnd uint32, pvid, untagged, self, master bool) error {
	base := link.Attrs()
	h.ensureIndex(base)
	req := h.newNetlinkRequest(cmd, unix.NLM_F_ACK)

	msg := nl.NewIfInfomsg(unix.AF_BRIDGE)
	msg.Index = int32(base.Index)
	req.AddData(msg)

	br := nl.NewRtAttr(unix.IFLA_AF_SPEC, nil)
	var flags uint16
	if self {
		flags |= nl.BRIDGE_FLAGS_SELF
	}
	if master {
		flags |= nl.BRIDGE_FLAGS_MASTER
	}
	if flags > 0 {
		br.AddRtAttr(nl.IFLA_BRIDGE_FLAGS, nl.Uint16Attr(flags))
	}

	if tunid != 0 {
		if tunidEnd != 0 {
			tiStart := br.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_INFO, nil)
			tiStart.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_ID, nl.Uint32Attr(tunid))
			tiStart.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_VID, nl.Uint16Attr(vid))
			tiStart.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_FLAGS, nl.Uint16Attr(nl.BRIDGE_VLAN_INFO_RANGE_BEGIN))

			tiEnd := br.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_INFO, nil)
			tiEnd.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_ID, nl.Uint32Attr(tunidEnd))
			tiEnd.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_VID, nl.Uint16Attr(vidEnd))
			tiEnd.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_FLAGS, nl.Uint16Attr(nl.BRIDGE_VLAN_INFO_RANGE_END))
		} else {
			ti := br.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_INFO, nil)
			ti.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_ID, nl.Uint32Attr(tunid))
			ti.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_VID, nl.Uint16Attr(vid))
			ti.AddRtAttr(nl.IFLA_BRIDGE_VLAN_TUNNEL_FLAGS, nl.Uint16Attr(0))
		}
	} else {
		vlanInfo := &nl.BridgeVlanInfo{Vid: vid}
		if pvid {
			vlanInfo.Flags |= nl.BRIDGE_VLAN_INFO_PVID
		}
		if untagged {
			vlanInfo.Flags |= nl.BRIDGE_VLAN_INFO_UNTAGGED
		}

		if vidEnd != 0 {
			vlanEndInfo := &nl.BridgeVlanInfo{Vid: vidEnd}
			vlanEndInfo.Flags = vlanInfo.Flags

			vlanInfo.Flags |= nl.BRIDGE_VLAN_INFO_RANGE_BEGIN
			br.AddRtAttr(nl.IFLA_BRIDGE_VLAN_INFO, vlanInfo.Serialize())

			vlanEndInfo.Flags |= nl.BRIDGE_VLAN_INFO_RANGE_END
			br.AddRtAttr(nl.IFLA_BRIDGE_VLAN_INFO, vlanEndInfo.Serialize())
		} else {
			br.AddRtAttr(nl.IFLA_BRIDGE_VLAN_INFO, vlanInfo.Serialize())
		}
	}

	req.AddData(br)
	_, err := req.Execute(unix.NETLINK_ROUTE, 0)
	return err
}
