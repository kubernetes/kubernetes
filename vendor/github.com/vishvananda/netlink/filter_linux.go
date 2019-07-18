package netlink

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"syscall"
	"unsafe"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

// Constants used in TcU32Sel.Flags.
const (
	TC_U32_TERMINAL  = nl.TC_U32_TERMINAL
	TC_U32_OFFSET    = nl.TC_U32_OFFSET
	TC_U32_VAROFFSET = nl.TC_U32_VAROFFSET
	TC_U32_EAT       = nl.TC_U32_EAT
)

// Fw filter filters on firewall marks
// NOTE: this is in filter_linux because it refers to nl.TcPolice which
//       is defined in nl/tc_linux.go
type Fw struct {
	FilterAttrs
	ClassId uint32
	// TODO remove nl type from interface
	Police nl.TcPolice
	InDev  string
	// TODO Action
	Mask   uint32
	AvRate uint32
	Rtab   [256]uint32
	Ptab   [256]uint32
}

func NewFw(attrs FilterAttrs, fattrs FilterFwAttrs) (*Fw, error) {
	var rtab [256]uint32
	var ptab [256]uint32
	rcellLog := -1
	pcellLog := -1
	avrate := fattrs.AvRate / 8
	police := nl.TcPolice{}
	police.Rate.Rate = fattrs.Rate / 8
	police.PeakRate.Rate = fattrs.PeakRate / 8
	buffer := fattrs.Buffer
	linklayer := nl.LINKLAYER_ETHERNET

	if fattrs.LinkLayer != nl.LINKLAYER_UNSPEC {
		linklayer = fattrs.LinkLayer
	}

	police.Action = int32(fattrs.Action)
	if police.Rate.Rate != 0 {
		police.Rate.Mpu = fattrs.Mpu
		police.Rate.Overhead = fattrs.Overhead
		if CalcRtable(&police.Rate, rtab[:], rcellLog, fattrs.Mtu, linklayer) < 0 {
			return nil, errors.New("TBF: failed to calculate rate table")
		}
		police.Burst = uint32(Xmittime(uint64(police.Rate.Rate), uint32(buffer)))
	}
	police.Mtu = fattrs.Mtu
	if police.PeakRate.Rate != 0 {
		police.PeakRate.Mpu = fattrs.Mpu
		police.PeakRate.Overhead = fattrs.Overhead
		if CalcRtable(&police.PeakRate, ptab[:], pcellLog, fattrs.Mtu, linklayer) < 0 {
			return nil, errors.New("POLICE: failed to calculate peak rate table")
		}
	}

	return &Fw{
		FilterAttrs: attrs,
		ClassId:     fattrs.ClassId,
		InDev:       fattrs.InDev,
		Mask:        fattrs.Mask,
		Police:      police,
		AvRate:      avrate,
		Rtab:        rtab,
		Ptab:        ptab,
	}, nil
}

func (filter *Fw) Attrs() *FilterAttrs {
	return &filter.FilterAttrs
}

func (filter *Fw) Type() string {
	return "fw"
}

// FilterDel will delete a filter from the system.
// Equivalent to: `tc filter del $filter`
func FilterDel(filter Filter) error {
	return pkgHandle.FilterDel(filter)
}

// FilterDel will delete a filter from the system.
// Equivalent to: `tc filter del $filter`
func (h *Handle) FilterDel(filter Filter) error {
	req := h.newNetlinkRequest(unix.RTM_DELTFILTER, unix.NLM_F_ACK)
	base := filter.Attrs()
	msg := &nl.TcMsg{
		Family:  nl.FAMILY_ALL,
		Ifindex: int32(base.LinkIndex),
		Handle:  base.Handle,
		Parent:  base.Parent,
		Info:    MakeHandle(base.Priority, nl.Swap16(base.Protocol)),
	}
	req.AddData(msg)

	_, err := req.Execute(unix.NETLINK_ROUTE, 0)
	return err
}

// FilterAdd will add a filter to the system.
// Equivalent to: `tc filter add $filter`
func FilterAdd(filter Filter) error {
	return pkgHandle.FilterAdd(filter)
}

// FilterAdd will add a filter to the system.
// Equivalent to: `tc filter add $filter`
func (h *Handle) FilterAdd(filter Filter) error {
	native = nl.NativeEndian()
	req := h.newNetlinkRequest(unix.RTM_NEWTFILTER, unix.NLM_F_CREATE|unix.NLM_F_EXCL|unix.NLM_F_ACK)
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

	switch filter := filter.(type) {
	case *U32:
		// Convert TcU32Sel into nl.TcU32Sel as it is without copy.
		sel := (*nl.TcU32Sel)(unsafe.Pointer(filter.Sel))
		if sel == nil {
			// match all
			sel = &nl.TcU32Sel{
				Nkeys: 1,
				Flags: nl.TC_U32_TERMINAL,
			}
			sel.Keys = append(sel.Keys, nl.TcU32Key{})
		}

		if native != networkOrder {
			// Copy TcU32Sel.
			cSel := *sel
			keys := make([]nl.TcU32Key, cap(sel.Keys))
			copy(keys, sel.Keys)
			cSel.Keys = keys
			sel = &cSel

			// Handle the endianness of attributes
			sel.Offmask = native.Uint16(htons(sel.Offmask))
			sel.Hmask = native.Uint32(htonl(sel.Hmask))
			for i, key := range sel.Keys {
				sel.Keys[i].Mask = native.Uint32(htonl(key.Mask))
				sel.Keys[i].Val = native.Uint32(htonl(key.Val))
			}
		}
		sel.Nkeys = uint8(len(sel.Keys))
		nl.NewRtAttrChild(options, nl.TCA_U32_SEL, sel.Serialize())
		if filter.ClassId != 0 {
			nl.NewRtAttrChild(options, nl.TCA_U32_CLASSID, nl.Uint32Attr(filter.ClassId))
		}
		actionsAttr := nl.NewRtAttrChild(options, nl.TCA_U32_ACT, nil)
		// backwards compatibility
		if filter.RedirIndex != 0 {
			filter.Actions = append([]Action{NewMirredAction(filter.RedirIndex)}, filter.Actions...)
		}
		if err := EncodeActions(actionsAttr, filter.Actions); err != nil {
			return err
		}
	case *Fw:
		if filter.Mask != 0 {
			b := make([]byte, 4)
			native.PutUint32(b, filter.Mask)
			nl.NewRtAttrChild(options, nl.TCA_FW_MASK, b)
		}
		if filter.InDev != "" {
			nl.NewRtAttrChild(options, nl.TCA_FW_INDEV, nl.ZeroTerminated(filter.InDev))
		}
		if (filter.Police != nl.TcPolice{}) {

			police := nl.NewRtAttrChild(options, nl.TCA_FW_POLICE, nil)
			nl.NewRtAttrChild(police, nl.TCA_POLICE_TBF, filter.Police.Serialize())
			if (filter.Police.Rate != nl.TcRateSpec{}) {
				payload := SerializeRtab(filter.Rtab)
				nl.NewRtAttrChild(police, nl.TCA_POLICE_RATE, payload)
			}
			if (filter.Police.PeakRate != nl.TcRateSpec{}) {
				payload := SerializeRtab(filter.Ptab)
				nl.NewRtAttrChild(police, nl.TCA_POLICE_PEAKRATE, payload)
			}
		}
		if filter.ClassId != 0 {
			b := make([]byte, 4)
			native.PutUint32(b, filter.ClassId)
			nl.NewRtAttrChild(options, nl.TCA_FW_CLASSID, b)
		}
	case *BpfFilter:
		var bpfFlags uint32
		if filter.ClassId != 0 {
			nl.NewRtAttrChild(options, nl.TCA_BPF_CLASSID, nl.Uint32Attr(filter.ClassId))
		}
		if filter.Fd >= 0 {
			nl.NewRtAttrChild(options, nl.TCA_BPF_FD, nl.Uint32Attr((uint32(filter.Fd))))
		}
		if filter.Name != "" {
			nl.NewRtAttrChild(options, nl.TCA_BPF_NAME, nl.ZeroTerminated(filter.Name))
		}
		if filter.DirectAction {
			bpfFlags |= nl.TCA_BPF_FLAG_ACT_DIRECT
		}
		nl.NewRtAttrChild(options, nl.TCA_BPF_FLAGS, nl.Uint32Attr(bpfFlags))
	case *MatchAll:
		actionsAttr := nl.NewRtAttrChild(options, nl.TCA_MATCHALL_ACT, nil)
		if err := EncodeActions(actionsAttr, filter.Actions); err != nil {
			return err
		}
		if filter.ClassId != 0 {
			nl.NewRtAttrChild(options, nl.TCA_MATCHALL_CLASSID, nl.Uint32Attr(filter.ClassId))
		}
	}

	req.AddData(options)
	_, err := req.Execute(unix.NETLINK_ROUTE, 0)
	return err
}

// FilterList gets a list of filters in the system.
// Equivalent to: `tc filter show`.
// Generally returns nothing if link and parent are not specified.
func FilterList(link Link, parent uint32) ([]Filter, error) {
	return pkgHandle.FilterList(link, parent)
}

// FilterList gets a list of filters in the system.
// Equivalent to: `tc filter show`.
// Generally returns nothing if link and parent are not specified.
func (h *Handle) FilterList(link Link, parent uint32) ([]Filter, error) {
	req := h.newNetlinkRequest(unix.RTM_GETTFILTER, unix.NLM_F_DUMP)
	msg := &nl.TcMsg{
		Family: nl.FAMILY_ALL,
		Parent: parent,
	}
	if link != nil {
		base := link.Attrs()
		h.ensureIndex(base)
		msg.Ifindex = int32(base.Index)
	}
	req.AddData(msg)

	msgs, err := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWTFILTER)
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
				case "fw":
					filter = &Fw{}
				case "bpf":
					filter = &BpfFilter{}
				case "matchall":
					filter = &MatchAll{}
				default:
					filter = &GenericFilter{FilterType: filterType}
				}
			case nl.TCA_OPTIONS:
				data, err := nl.ParseRouteAttr(attr.Value)
				if err != nil {
					return nil, err
				}
				switch filterType {
				case "u32":
					detailed, err = parseU32Data(filter, data)
					if err != nil {
						return nil, err
					}
				case "fw":
					detailed, err = parseFwData(filter, data)
					if err != nil {
						return nil, err
					}
				case "bpf":
					detailed, err = parseBpfData(filter, data)
					if err != nil {
						return nil, err
					}
				case "matchall":
					detailed, err = parseMatchAllData(filter, data)
					if err != nil {
						return nil, err
					}
				default:
					detailed = true
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

func toTcGen(attrs *ActionAttrs, tcgen *nl.TcGen) {
	tcgen.Index = uint32(attrs.Index)
	tcgen.Capab = uint32(attrs.Capab)
	tcgen.Action = int32(attrs.Action)
	tcgen.Refcnt = int32(attrs.Refcnt)
	tcgen.Bindcnt = int32(attrs.Bindcnt)
}

func toAttrs(tcgen *nl.TcGen, attrs *ActionAttrs) {
	attrs.Index = int(tcgen.Index)
	attrs.Capab = int(tcgen.Capab)
	attrs.Action = TcAct(tcgen.Action)
	attrs.Refcnt = int(tcgen.Refcnt)
	attrs.Bindcnt = int(tcgen.Bindcnt)
}

func EncodeActions(attr *nl.RtAttr, actions []Action) error {
	tabIndex := int(nl.TCA_ACT_TAB)

	for _, action := range actions {
		switch action := action.(type) {
		default:
			return fmt.Errorf("unknown action type %s", action.Type())
		case *MirredAction:
			table := nl.NewRtAttrChild(attr, tabIndex, nil)
			tabIndex++
			nl.NewRtAttrChild(table, nl.TCA_ACT_KIND, nl.ZeroTerminated("mirred"))
			aopts := nl.NewRtAttrChild(table, nl.TCA_ACT_OPTIONS, nil)
			mirred := nl.TcMirred{
				Eaction: int32(action.MirredAction),
				Ifindex: uint32(action.Ifindex),
			}
			toTcGen(action.Attrs(), &mirred.TcGen)
			nl.NewRtAttrChild(aopts, nl.TCA_MIRRED_PARMS, mirred.Serialize())
		case *BpfAction:
			table := nl.NewRtAttrChild(attr, tabIndex, nil)
			tabIndex++
			nl.NewRtAttrChild(table, nl.TCA_ACT_KIND, nl.ZeroTerminated("bpf"))
			aopts := nl.NewRtAttrChild(table, nl.TCA_ACT_OPTIONS, nil)
			gen := nl.TcGen{}
			toTcGen(action.Attrs(), &gen)
			nl.NewRtAttrChild(aopts, nl.TCA_ACT_BPF_PARMS, gen.Serialize())
			nl.NewRtAttrChild(aopts, nl.TCA_ACT_BPF_FD, nl.Uint32Attr(uint32(action.Fd)))
			nl.NewRtAttrChild(aopts, nl.TCA_ACT_BPF_NAME, nl.ZeroTerminated(action.Name))
		case *GenericAction:
			table := nl.NewRtAttrChild(attr, tabIndex, nil)
			tabIndex++
			nl.NewRtAttrChild(table, nl.TCA_ACT_KIND, nl.ZeroTerminated("gact"))
			aopts := nl.NewRtAttrChild(table, nl.TCA_ACT_OPTIONS, nil)
			gen := nl.TcGen{}
			toTcGen(action.Attrs(), &gen)
			nl.NewRtAttrChild(aopts, nl.TCA_GACT_PARMS, gen.Serialize())
		}
	}
	return nil
}

func parseActions(tables []syscall.NetlinkRouteAttr) ([]Action, error) {
	var actions []Action
	for _, table := range tables {
		var action Action
		var actionType string
		aattrs, err := nl.ParseRouteAttr(table.Value)
		if err != nil {
			return nil, err
		}
	nextattr:
		for _, aattr := range aattrs {
			switch aattr.Attr.Type {
			case nl.TCA_KIND:
				actionType = string(aattr.Value[:len(aattr.Value)-1])
				// only parse if the action is mirred or bpf
				switch actionType {
				case "mirred":
					action = &MirredAction{}
				case "bpf":
					action = &BpfAction{}
				case "gact":
					action = &GenericAction{}
				default:
					break nextattr
				}
			case nl.TCA_OPTIONS:
				adata, err := nl.ParseRouteAttr(aattr.Value)
				if err != nil {
					return nil, err
				}
				for _, adatum := range adata {
					switch actionType {
					case "mirred":
						switch adatum.Attr.Type {
						case nl.TCA_MIRRED_PARMS:
							mirred := *nl.DeserializeTcMirred(adatum.Value)
							toAttrs(&mirred.TcGen, action.Attrs())
							action.(*MirredAction).ActionAttrs = ActionAttrs{}
							action.(*MirredAction).Ifindex = int(mirred.Ifindex)
							action.(*MirredAction).MirredAction = MirredAct(mirred.Eaction)
						}
					case "bpf":
						switch adatum.Attr.Type {
						case nl.TCA_ACT_BPF_PARMS:
							gen := *nl.DeserializeTcGen(adatum.Value)
							toAttrs(&gen, action.Attrs())
						case nl.TCA_ACT_BPF_FD:
							action.(*BpfAction).Fd = int(native.Uint32(adatum.Value[0:4]))
						case nl.TCA_ACT_BPF_NAME:
							action.(*BpfAction).Name = string(adatum.Value[:len(adatum.Value)-1])
						}
					case "gact":
						switch adatum.Attr.Type {
						case nl.TCA_GACT_PARMS:
							gen := *nl.DeserializeTcGen(adatum.Value)
							toAttrs(&gen, action.Attrs())
						}
					}
				}
			}
		}
		actions = append(actions, action)
	}
	return actions, nil
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
			u32.Sel = (*TcU32Sel)(unsafe.Pointer(sel))
			if native != networkOrder {
				// Handle the endianness of attributes
				u32.Sel.Offmask = native.Uint16(htons(sel.Offmask))
				u32.Sel.Hmask = native.Uint32(htonl(sel.Hmask))
				for i, key := range u32.Sel.Keys {
					u32.Sel.Keys[i].Mask = native.Uint32(htonl(key.Mask))
					u32.Sel.Keys[i].Val = native.Uint32(htonl(key.Val))
				}
			}
		case nl.TCA_U32_ACT:
			tables, err := nl.ParseRouteAttr(datum.Value)
			if err != nil {
				return detailed, err
			}
			u32.Actions, err = parseActions(tables)
			if err != nil {
				return detailed, err
			}
			for _, action := range u32.Actions {
				if action, ok := action.(*MirredAction); ok {
					u32.RedirIndex = int(action.Ifindex)
				}
			}
		case nl.TCA_U32_CLASSID:
			u32.ClassId = native.Uint32(datum.Value)
		}
	}
	return detailed, nil
}

func parseFwData(filter Filter, data []syscall.NetlinkRouteAttr) (bool, error) {
	native = nl.NativeEndian()
	fw := filter.(*Fw)
	detailed := true
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_FW_MASK:
			fw.Mask = native.Uint32(datum.Value[0:4])
		case nl.TCA_FW_CLASSID:
			fw.ClassId = native.Uint32(datum.Value[0:4])
		case nl.TCA_FW_INDEV:
			fw.InDev = string(datum.Value[:len(datum.Value)-1])
		case nl.TCA_FW_POLICE:
			adata, _ := nl.ParseRouteAttr(datum.Value)
			for _, aattr := range adata {
				switch aattr.Attr.Type {
				case nl.TCA_POLICE_TBF:
					fw.Police = *nl.DeserializeTcPolice(aattr.Value)
				case nl.TCA_POLICE_RATE:
					fw.Rtab = DeserializeRtab(aattr.Value)
				case nl.TCA_POLICE_PEAKRATE:
					fw.Ptab = DeserializeRtab(aattr.Value)
				}
			}
		}
	}
	return detailed, nil
}

func parseBpfData(filter Filter, data []syscall.NetlinkRouteAttr) (bool, error) {
	native = nl.NativeEndian()
	bpf := filter.(*BpfFilter)
	detailed := true
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_BPF_FD:
			bpf.Fd = int(native.Uint32(datum.Value[0:4]))
		case nl.TCA_BPF_NAME:
			bpf.Name = string(datum.Value[:len(datum.Value)-1])
		case nl.TCA_BPF_CLASSID:
			bpf.ClassId = native.Uint32(datum.Value[0:4])
		case nl.TCA_BPF_FLAGS:
			flags := native.Uint32(datum.Value[0:4])
			if (flags & nl.TCA_BPF_FLAG_ACT_DIRECT) != 0 {
				bpf.DirectAction = true
			}
		}
	}
	return detailed, nil
}

func parseMatchAllData(filter Filter, data []syscall.NetlinkRouteAttr) (bool, error) {
	native = nl.NativeEndian()
	matchall := filter.(*MatchAll)
	detailed := true
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_MATCHALL_CLASSID:
			matchall.ClassId = native.Uint32(datum.Value[0:4])
		case nl.TCA_MATCHALL_ACT:
			tables, err := nl.ParseRouteAttr(datum.Value)
			if err != nil {
				return detailed, err
			}
			matchall.Actions, err = parseActions(tables)
			if err != nil {
				return detailed, err
			}
		}
	}
	return detailed, nil
}

func AlignToAtm(size uint) uint {
	var linksize, cells int
	cells = int(size / nl.ATM_CELL_PAYLOAD)
	if (size % nl.ATM_CELL_PAYLOAD) > 0 {
		cells++
	}
	linksize = cells * nl.ATM_CELL_SIZE
	return uint(linksize)
}

func AdjustSize(sz uint, mpu uint, linklayer int) uint {
	if sz < mpu {
		sz = mpu
	}
	switch linklayer {
	case nl.LINKLAYER_ATM:
		return AlignToAtm(sz)
	default:
		return sz
	}
}

func CalcRtable(rate *nl.TcRateSpec, rtab []uint32, cellLog int, mtu uint32, linklayer int) int {
	bps := rate.Rate
	mpu := rate.Mpu
	var sz uint
	if mtu == 0 {
		mtu = 2047
	}
	if cellLog < 0 {
		cellLog = 0
		for (mtu >> uint(cellLog)) > 255 {
			cellLog++
		}
	}
	for i := 0; i < 256; i++ {
		sz = AdjustSize(uint((i+1)<<uint32(cellLog)), uint(mpu), linklayer)
		rtab[i] = uint32(Xmittime(uint64(bps), uint32(sz)))
	}
	rate.CellAlign = -1
	rate.CellLog = uint8(cellLog)
	rate.Linklayer = uint8(linklayer & nl.TC_LINKLAYER_MASK)
	return cellLog
}

func DeserializeRtab(b []byte) [256]uint32 {
	var rtab [256]uint32
	native := nl.NativeEndian()
	r := bytes.NewReader(b)
	_ = binary.Read(r, native, &rtab)
	return rtab
}

func SerializeRtab(rtab [256]uint32) []byte {
	native := nl.NativeEndian()
	var w bytes.Buffer
	_ = binary.Write(&w, native, rtab)
	return w.Bytes()
}
