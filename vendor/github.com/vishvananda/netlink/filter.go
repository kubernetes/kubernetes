package netlink

import (
	"fmt"
	"net"
)

type Filter interface {
	Attrs() *FilterAttrs
	Type() string
}

// FilterAttrs represents a netlink filter. A filter is associated with a link,
// has a handle and a parent. The root filter of a device should have a
// parent == HANDLE_ROOT.
type FilterAttrs struct {
	LinkIndex int
	Handle    uint32
	Parent    uint32
	Priority  uint16 // lower is higher priority
	Protocol  uint16 // unix.ETH_P_*
}

func (q FilterAttrs) String() string {
	return fmt.Sprintf("{LinkIndex: %d, Handle: %s, Parent: %s, Priority: %d, Protocol: %d}", q.LinkIndex, HandleStr(q.Handle), HandleStr(q.Parent), q.Priority, q.Protocol)
}

type TcAct int32

const (
	TC_ACT_UNSPEC     TcAct = -1
	TC_ACT_OK         TcAct = 0
	TC_ACT_RECLASSIFY TcAct = 1
	TC_ACT_SHOT       TcAct = 2
	TC_ACT_PIPE       TcAct = 3
	TC_ACT_STOLEN     TcAct = 4
	TC_ACT_QUEUED     TcAct = 5
	TC_ACT_REPEAT     TcAct = 6
	TC_ACT_REDIRECT   TcAct = 7
	TC_ACT_JUMP       TcAct = 0x10000000
)

func (a TcAct) String() string {
	switch a {
	case TC_ACT_UNSPEC:
		return "unspec"
	case TC_ACT_OK:
		return "ok"
	case TC_ACT_RECLASSIFY:
		return "reclassify"
	case TC_ACT_SHOT:
		return "shot"
	case TC_ACT_PIPE:
		return "pipe"
	case TC_ACT_STOLEN:
		return "stolen"
	case TC_ACT_QUEUED:
		return "queued"
	case TC_ACT_REPEAT:
		return "repeat"
	case TC_ACT_REDIRECT:
		return "redirect"
	case TC_ACT_JUMP:
		return "jump"
	}
	return fmt.Sprintf("0x%x", int32(a))
}

type TcPolAct int32

const (
	TC_POLICE_UNSPEC     TcPolAct = TcPolAct(TC_ACT_UNSPEC)
	TC_POLICE_OK         TcPolAct = TcPolAct(TC_ACT_OK)
	TC_POLICE_RECLASSIFY TcPolAct = TcPolAct(TC_ACT_RECLASSIFY)
	TC_POLICE_SHOT       TcPolAct = TcPolAct(TC_ACT_SHOT)
	TC_POLICE_PIPE       TcPolAct = TcPolAct(TC_ACT_PIPE)
)

func (a TcPolAct) String() string {
	switch a {
	case TC_POLICE_UNSPEC:
		return "unspec"
	case TC_POLICE_OK:
		return "ok"
	case TC_POLICE_RECLASSIFY:
		return "reclassify"
	case TC_POLICE_SHOT:
		return "shot"
	case TC_POLICE_PIPE:
		return "pipe"
	}
	return fmt.Sprintf("0x%x", int32(a))
}

type ActionAttrs struct {
	Index   int
	Capab   int
	Action  TcAct
	Refcnt  int
	Bindcnt int
}

func (q ActionAttrs) String() string {
	return fmt.Sprintf("{Index: %d, Capab: %x, Action: %s, Refcnt: %d, Bindcnt: %d}", q.Index, q.Capab, q.Action.String(), q.Refcnt, q.Bindcnt)
}

// Action represents an action in any supported filter.
type Action interface {
	Attrs() *ActionAttrs
	Type() string
}

type GenericAction struct {
	ActionAttrs
}

func (action *GenericAction) Type() string {
	return "generic"
}

func (action *GenericAction) Attrs() *ActionAttrs {
	return &action.ActionAttrs
}

type BpfAction struct {
	ActionAttrs
	Fd   int
	Name string
}

func (action *BpfAction) Type() string {
	return "bpf"
}

func (action *BpfAction) Attrs() *ActionAttrs {
	return &action.ActionAttrs
}

type ConnmarkAction struct {
	ActionAttrs
	Zone uint16
}

func (action *ConnmarkAction) Type() string {
	return "connmark"
}

func (action *ConnmarkAction) Attrs() *ActionAttrs {
	return &action.ActionAttrs
}

func NewConnmarkAction() *ConnmarkAction {
	return &ConnmarkAction{
		ActionAttrs: ActionAttrs{
			Action: TC_ACT_PIPE,
		},
	}
}

type MirredAct uint8

func (a MirredAct) String() string {
	switch a {
	case TCA_EGRESS_REDIR:
		return "egress redir"
	case TCA_EGRESS_MIRROR:
		return "egress mirror"
	case TCA_INGRESS_REDIR:
		return "ingress redir"
	case TCA_INGRESS_MIRROR:
		return "ingress mirror"
	}
	return "unknown"
}

const (
	TCA_EGRESS_REDIR   MirredAct = 1 /* packet redirect to EGRESS*/
	TCA_EGRESS_MIRROR  MirredAct = 2 /* mirror packet to EGRESS */
	TCA_INGRESS_REDIR  MirredAct = 3 /* packet redirect to INGRESS*/
	TCA_INGRESS_MIRROR MirredAct = 4 /* mirror packet to INGRESS */
)

type MirredAction struct {
	ActionAttrs
	MirredAction MirredAct
	Ifindex      int
}

func (action *MirredAction) Type() string {
	return "mirred"
}

func (action *MirredAction) Attrs() *ActionAttrs {
	return &action.ActionAttrs
}

func NewMirredAction(redirIndex int) *MirredAction {
	return &MirredAction{
		ActionAttrs: ActionAttrs{
			Action: TC_ACT_STOLEN,
		},
		MirredAction: TCA_EGRESS_REDIR,
		Ifindex:      redirIndex,
	}
}

type TunnelKeyAct int8

const (
	TCA_TUNNEL_KEY_SET   TunnelKeyAct = 1 // set tunnel key
	TCA_TUNNEL_KEY_UNSET TunnelKeyAct = 2 // unset tunnel key
)

type TunnelKeyAction struct {
	ActionAttrs
	Action  TunnelKeyAct
	SrcAddr net.IP
	DstAddr net.IP
	KeyID   uint32
}

func (action *TunnelKeyAction) Type() string {
	return "tunnel_key"
}

func (action *TunnelKeyAction) Attrs() *ActionAttrs {
	return &action.ActionAttrs
}

func NewTunnelKeyAction() *TunnelKeyAction {
	return &TunnelKeyAction{
		ActionAttrs: ActionAttrs{
			Action: TC_ACT_PIPE,
		},
	}
}

type SkbEditAction struct {
	ActionAttrs
	QueueMapping *uint16
	PType        *uint16
	Priority     *uint32
	Mark         *uint32
}

func (action *SkbEditAction) Type() string {
	return "skbedit"
}

func (action *SkbEditAction) Attrs() *ActionAttrs {
	return &action.ActionAttrs
}

func NewSkbEditAction() *SkbEditAction {
	return &SkbEditAction{
		ActionAttrs: ActionAttrs{
			Action: TC_ACT_PIPE,
		},
	}
}

// MatchAll filters match all packets
type MatchAll struct {
	FilterAttrs
	ClassId uint32
	Actions []Action
}

func (filter *MatchAll) Attrs() *FilterAttrs {
	return &filter.FilterAttrs
}

func (filter *MatchAll) Type() string {
	return "matchall"
}

type FilterFwAttrs struct {
	ClassId   uint32
	InDev     string
	Mask      uint32
	Index     uint32
	Buffer    uint32
	Mtu       uint32
	Mpu       uint16
	Rate      uint32
	AvRate    uint32
	PeakRate  uint32
	Action    TcPolAct
	Overhead  uint16
	LinkLayer int
}

type BpfFilter struct {
	FilterAttrs
	ClassId      uint32
	Fd           int
	Name         string
	DirectAction bool
	Id           int
	Tag          string
}

func (filter *BpfFilter) Type() string {
	return "bpf"
}

func (filter *BpfFilter) Attrs() *FilterAttrs {
	return &filter.FilterAttrs
}

// GenericFilter filters represent types that are not currently understood
// by this netlink library.
type GenericFilter struct {
	FilterAttrs
	FilterType string
}

func (filter *GenericFilter) Attrs() *FilterAttrs {
	return &filter.FilterAttrs
}

func (filter *GenericFilter) Type() string {
	return filter.FilterType
}
