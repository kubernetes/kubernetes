package netlink

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

// NOTE function is here because it uses other linux functions
func NewNetem(attrs QdiscAttrs, nattrs NetemQdiscAttrs) *Netem {
	var limit uint32 = 1000
	var lossCorr, delayCorr, duplicateCorr uint32
	var reorderProb, reorderCorr uint32
	var corruptProb, corruptCorr uint32

	latency := nattrs.Latency
	loss := Percentage2u32(nattrs.Loss)
	gap := nattrs.Gap
	duplicate := Percentage2u32(nattrs.Duplicate)
	jitter := nattrs.Jitter

	// Correlation
	if latency > 0 && jitter > 0 {
		delayCorr = Percentage2u32(nattrs.DelayCorr)
	}
	if loss > 0 {
		lossCorr = Percentage2u32(nattrs.LossCorr)
	}
	if duplicate > 0 {
		duplicateCorr = Percentage2u32(nattrs.DuplicateCorr)
	}
	// FIXME should validate values(like loss/duplicate are percentages...)
	latency = time2Tick(latency)

	if nattrs.Limit != 0 {
		limit = nattrs.Limit
	}
	// Jitter is only value if latency is > 0
	if latency > 0 {
		jitter = time2Tick(jitter)
	}

	reorderProb = Percentage2u32(nattrs.ReorderProb)
	reorderCorr = Percentage2u32(nattrs.ReorderCorr)

	if reorderProb > 0 {
		// ERROR if lantency == 0
		if gap == 0 {
			gap = 1
		}
	}

	corruptProb = Percentage2u32(nattrs.CorruptProb)
	corruptCorr = Percentage2u32(nattrs.CorruptCorr)

	return &Netem{
		QdiscAttrs:    attrs,
		Latency:       latency,
		DelayCorr:     delayCorr,
		Limit:         limit,
		Loss:          loss,
		LossCorr:      lossCorr,
		Gap:           gap,
		Duplicate:     duplicate,
		DuplicateCorr: duplicateCorr,
		Jitter:        jitter,
		ReorderProb:   reorderProb,
		ReorderCorr:   reorderCorr,
		CorruptProb:   corruptProb,
		CorruptCorr:   corruptCorr,
	}
}

// QdiscDel will delete a qdisc from the system.
// Equivalent to: `tc qdisc del $qdisc`
func QdiscDel(qdisc Qdisc) error {
	return pkgHandle.QdiscDel(qdisc)
}

// QdiscDel will delete a qdisc from the system.
// Equivalent to: `tc qdisc del $qdisc`
func (h *Handle) QdiscDel(qdisc Qdisc) error {
	return h.qdiscModify(unix.RTM_DELQDISC, 0, qdisc)
}

// QdiscChange will change a qdisc in place
// Equivalent to: `tc qdisc change $qdisc`
// The parent and handle MUST NOT be changed.
func QdiscChange(qdisc Qdisc) error {
	return pkgHandle.QdiscChange(qdisc)
}

// QdiscChange will change a qdisc in place
// Equivalent to: `tc qdisc change $qdisc`
// The parent and handle MUST NOT be changed.
func (h *Handle) QdiscChange(qdisc Qdisc) error {
	return h.qdiscModify(unix.RTM_NEWQDISC, 0, qdisc)
}

// QdiscReplace will replace a qdisc to the system.
// Equivalent to: `tc qdisc replace $qdisc`
// The handle MUST change.
func QdiscReplace(qdisc Qdisc) error {
	return pkgHandle.QdiscReplace(qdisc)
}

// QdiscReplace will replace a qdisc to the system.
// Equivalent to: `tc qdisc replace $qdisc`
// The handle MUST change.
func (h *Handle) QdiscReplace(qdisc Qdisc) error {
	return h.qdiscModify(
		unix.RTM_NEWQDISC,
		unix.NLM_F_CREATE|unix.NLM_F_REPLACE,
		qdisc)
}

// QdiscAdd will add a qdisc to the system.
// Equivalent to: `tc qdisc add $qdisc`
func QdiscAdd(qdisc Qdisc) error {
	return pkgHandle.QdiscAdd(qdisc)
}

// QdiscAdd will add a qdisc to the system.
// Equivalent to: `tc qdisc add $qdisc`
func (h *Handle) QdiscAdd(qdisc Qdisc) error {
	return h.qdiscModify(
		unix.RTM_NEWQDISC,
		unix.NLM_F_CREATE|unix.NLM_F_EXCL,
		qdisc)
}

func (h *Handle) qdiscModify(cmd, flags int, qdisc Qdisc) error {
	req := h.newNetlinkRequest(cmd, flags|unix.NLM_F_ACK)
	base := qdisc.Attrs()
	msg := &nl.TcMsg{
		Family:  nl.FAMILY_ALL,
		Ifindex: int32(base.LinkIndex),
		Handle:  base.Handle,
		Parent:  base.Parent,
	}
	req.AddData(msg)

	// When deleting don't bother building the rest of the netlink payload
	if cmd != unix.RTM_DELQDISC {
		if err := qdiscPayload(req, qdisc); err != nil {
			return err
		}
	}

	_, err := req.Execute(unix.NETLINK_ROUTE, 0)
	return err
}

func qdiscPayload(req *nl.NetlinkRequest, qdisc Qdisc) error {

	req.AddData(nl.NewRtAttr(nl.TCA_KIND, nl.ZeroTerminated(qdisc.Type())))

	options := nl.NewRtAttr(nl.TCA_OPTIONS, nil)

	switch qdisc := qdisc.(type) {
	case *Prio:
		tcmap := nl.TcPrioMap{
			Bands:   int32(qdisc.Bands),
			Priomap: qdisc.PriorityMap,
		}
		options = nl.NewRtAttr(nl.TCA_OPTIONS, tcmap.Serialize())
	case *Tbf:
		opt := nl.TcTbfQopt{}
		opt.Rate.Rate = uint32(qdisc.Rate)
		opt.Peakrate.Rate = uint32(qdisc.Peakrate)
		opt.Limit = qdisc.Limit
		opt.Buffer = qdisc.Buffer
		nl.NewRtAttrChild(options, nl.TCA_TBF_PARMS, opt.Serialize())
		if qdisc.Rate >= uint64(1<<32) {
			nl.NewRtAttrChild(options, nl.TCA_TBF_RATE64, nl.Uint64Attr(qdisc.Rate))
		}
		if qdisc.Peakrate >= uint64(1<<32) {
			nl.NewRtAttrChild(options, nl.TCA_TBF_PRATE64, nl.Uint64Attr(qdisc.Peakrate))
		}
		if qdisc.Peakrate > 0 {
			nl.NewRtAttrChild(options, nl.TCA_TBF_PBURST, nl.Uint32Attr(qdisc.Minburst))
		}
	case *Htb:
		opt := nl.TcHtbGlob{}
		opt.Version = qdisc.Version
		opt.Rate2Quantum = qdisc.Rate2Quantum
		opt.Defcls = qdisc.Defcls
		// TODO: Handle Debug properly. For now default to 0
		opt.Debug = qdisc.Debug
		opt.DirectPkts = qdisc.DirectPkts
		nl.NewRtAttrChild(options, nl.TCA_HTB_INIT, opt.Serialize())
		// nl.NewRtAttrChild(options, nl.TCA_HTB_DIRECT_QLEN, opt.Serialize())
	case *Netem:
		opt := nl.TcNetemQopt{}
		opt.Latency = qdisc.Latency
		opt.Limit = qdisc.Limit
		opt.Loss = qdisc.Loss
		opt.Gap = qdisc.Gap
		opt.Duplicate = qdisc.Duplicate
		opt.Jitter = qdisc.Jitter
		options = nl.NewRtAttr(nl.TCA_OPTIONS, opt.Serialize())
		// Correlation
		corr := nl.TcNetemCorr{}
		corr.DelayCorr = qdisc.DelayCorr
		corr.LossCorr = qdisc.LossCorr
		corr.DupCorr = qdisc.DuplicateCorr

		if corr.DelayCorr > 0 || corr.LossCorr > 0 || corr.DupCorr > 0 {
			nl.NewRtAttrChild(options, nl.TCA_NETEM_CORR, corr.Serialize())
		}
		// Corruption
		corruption := nl.TcNetemCorrupt{}
		corruption.Probability = qdisc.CorruptProb
		corruption.Correlation = qdisc.CorruptCorr
		if corruption.Probability > 0 {
			nl.NewRtAttrChild(options, nl.TCA_NETEM_CORRUPT, corruption.Serialize())
		}
		// Reorder
		reorder := nl.TcNetemReorder{}
		reorder.Probability = qdisc.ReorderProb
		reorder.Correlation = qdisc.ReorderCorr
		if reorder.Probability > 0 {
			nl.NewRtAttrChild(options, nl.TCA_NETEM_REORDER, reorder.Serialize())
		}
	case *Ingress:
		// ingress filters must use the proper handle
		if qdisc.Attrs().Parent != HANDLE_INGRESS {
			return fmt.Errorf("Ingress filters must set Parent to HANDLE_INGRESS")
		}
	case *FqCodel:
		nl.NewRtAttrChild(options, nl.TCA_FQ_CODEL_ECN, nl.Uint32Attr((uint32(qdisc.ECN))))
		if qdisc.Limit > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_CODEL_LIMIT, nl.Uint32Attr((uint32(qdisc.Limit))))
		}
		if qdisc.Interval > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_CODEL_INTERVAL, nl.Uint32Attr((uint32(qdisc.Interval))))
		}
		if qdisc.Flows > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_CODEL_FLOWS, nl.Uint32Attr((uint32(qdisc.Flows))))
		}
		if qdisc.Quantum > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_CODEL_QUANTUM, nl.Uint32Attr((uint32(qdisc.Quantum))))
		}

	case *Fq:
		nl.NewRtAttrChild(options, nl.TCA_FQ_RATE_ENABLE, nl.Uint32Attr((uint32(qdisc.Pacing))))

		if qdisc.Buckets > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_BUCKETS_LOG, nl.Uint32Attr((uint32(qdisc.Buckets))))
		}
		if qdisc.LowRateThreshold > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_LOW_RATE_THRESHOLD, nl.Uint32Attr((uint32(qdisc.LowRateThreshold))))
		}
		if qdisc.Quantum > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_QUANTUM, nl.Uint32Attr((uint32(qdisc.Quantum))))
		}
		if qdisc.InitialQuantum > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_INITIAL_QUANTUM, nl.Uint32Attr((uint32(qdisc.InitialQuantum))))
		}
		if qdisc.FlowRefillDelay > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_FLOW_REFILL_DELAY, nl.Uint32Attr((uint32(qdisc.FlowRefillDelay))))
		}
		if qdisc.FlowPacketLimit > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_FLOW_PLIMIT, nl.Uint32Attr((uint32(qdisc.FlowPacketLimit))))
		}
		if qdisc.FlowMaxRate > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_FLOW_MAX_RATE, nl.Uint32Attr((uint32(qdisc.FlowMaxRate))))
		}
		if qdisc.FlowDefaultRate > 0 {
			nl.NewRtAttrChild(options, nl.TCA_FQ_FLOW_DEFAULT_RATE, nl.Uint32Attr((uint32(qdisc.FlowDefaultRate))))
		}
	}

	req.AddData(options)
	return nil
}

// QdiscList gets a list of qdiscs in the system.
// Equivalent to: `tc qdisc show`.
// The list can be filtered by link.
func QdiscList(link Link) ([]Qdisc, error) {
	return pkgHandle.QdiscList(link)
}

// QdiscList gets a list of qdiscs in the system.
// Equivalent to: `tc qdisc show`.
// The list can be filtered by link.
func (h *Handle) QdiscList(link Link) ([]Qdisc, error) {
	req := h.newNetlinkRequest(unix.RTM_GETQDISC, unix.NLM_F_DUMP)
	index := int32(0)
	if link != nil {
		base := link.Attrs()
		h.ensureIndex(base)
		index = int32(base.Index)
	}
	msg := &nl.TcMsg{
		Family:  nl.FAMILY_ALL,
		Ifindex: index,
	}
	req.AddData(msg)

	msgs, err := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWQDISC)
	if err != nil {
		return nil, err
	}

	var res []Qdisc
	for _, m := range msgs {
		msg := nl.DeserializeTcMsg(m)

		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return nil, err
		}

		// skip qdiscs from other interfaces
		if link != nil && msg.Ifindex != index {
			continue
		}

		base := QdiscAttrs{
			LinkIndex: int(msg.Ifindex),
			Handle:    msg.Handle,
			Parent:    msg.Parent,
			Refcnt:    msg.Info,
		}
		var qdisc Qdisc
		qdiscType := ""
		for _, attr := range attrs {
			switch attr.Attr.Type {
			case nl.TCA_KIND:
				qdiscType = string(attr.Value[:len(attr.Value)-1])
				switch qdiscType {
				case "pfifo_fast":
					qdisc = &PfifoFast{}
				case "prio":
					qdisc = &Prio{}
				case "tbf":
					qdisc = &Tbf{}
				case "ingress":
					qdisc = &Ingress{}
				case "htb":
					qdisc = &Htb{}
				case "fq":
					qdisc = &Fq{}
				case "fq_codel":
					qdisc = &FqCodel{}
				case "netem":
					qdisc = &Netem{}
				default:
					qdisc = &GenericQdisc{QdiscType: qdiscType}
				}
			case nl.TCA_OPTIONS:
				switch qdiscType {
				case "pfifo_fast":
					// pfifo returns TcPrioMap directly without wrapping it in rtattr
					if err := parsePfifoFastData(qdisc, attr.Value); err != nil {
						return nil, err
					}
				case "prio":
					// prio returns TcPrioMap directly without wrapping it in rtattr
					if err := parsePrioData(qdisc, attr.Value); err != nil {
						return nil, err
					}
				case "tbf":
					data, err := nl.ParseRouteAttr(attr.Value)
					if err != nil {
						return nil, err
					}
					if err := parseTbfData(qdisc, data); err != nil {
						return nil, err
					}
				case "htb":
					data, err := nl.ParseRouteAttr(attr.Value)
					if err != nil {
						return nil, err
					}
					if err := parseHtbData(qdisc, data); err != nil {
						return nil, err
					}
				case "fq":
					data, err := nl.ParseRouteAttr(attr.Value)
					if err != nil {
						return nil, err
					}
					if err := parseFqData(qdisc, data); err != nil {
						return nil, err
					}
				case "fq_codel":
					data, err := nl.ParseRouteAttr(attr.Value)
					if err != nil {
						return nil, err
					}
					if err := parseFqCodelData(qdisc, data); err != nil {
						return nil, err
					}
				case "netem":
					if err := parseNetemData(qdisc, attr.Value); err != nil {
						return nil, err
					}

					// no options for ingress
				}
			}
		}
		*qdisc.Attrs() = base
		res = append(res, qdisc)
	}

	return res, nil
}

func parsePfifoFastData(qdisc Qdisc, value []byte) error {
	pfifo := qdisc.(*PfifoFast)
	tcmap := nl.DeserializeTcPrioMap(value)
	pfifo.PriorityMap = tcmap.Priomap
	pfifo.Bands = uint8(tcmap.Bands)
	return nil
}

func parsePrioData(qdisc Qdisc, value []byte) error {
	prio := qdisc.(*Prio)
	tcmap := nl.DeserializeTcPrioMap(value)
	prio.PriorityMap = tcmap.Priomap
	prio.Bands = uint8(tcmap.Bands)
	return nil
}

func parseHtbData(qdisc Qdisc, data []syscall.NetlinkRouteAttr) error {
	native = nl.NativeEndian()
	htb := qdisc.(*Htb)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_HTB_INIT:
			opt := nl.DeserializeTcHtbGlob(datum.Value)
			htb.Version = opt.Version
			htb.Rate2Quantum = opt.Rate2Quantum
			htb.Defcls = opt.Defcls
			htb.Debug = opt.Debug
			htb.DirectPkts = opt.DirectPkts
		case nl.TCA_HTB_DIRECT_QLEN:
			// TODO
			//htb.DirectQlen = native.uint32(datum.Value)
		}
	}
	return nil
}

func parseFqCodelData(qdisc Qdisc, data []syscall.NetlinkRouteAttr) error {
	native = nl.NativeEndian()
	fqCodel := qdisc.(*FqCodel)
	for _, datum := range data {

		switch datum.Attr.Type {
		case nl.TCA_FQ_CODEL_TARGET:
			fqCodel.Target = native.Uint32(datum.Value)
		case nl.TCA_FQ_CODEL_LIMIT:
			fqCodel.Limit = native.Uint32(datum.Value)
		case nl.TCA_FQ_CODEL_INTERVAL:
			fqCodel.Interval = native.Uint32(datum.Value)
		case nl.TCA_FQ_CODEL_ECN:
			fqCodel.ECN = native.Uint32(datum.Value)
		case nl.TCA_FQ_CODEL_FLOWS:
			fqCodel.Flows = native.Uint32(datum.Value)
		case nl.TCA_FQ_CODEL_QUANTUM:
			fqCodel.Quantum = native.Uint32(datum.Value)
		}
	}
	return nil
}

func parseFqData(qdisc Qdisc, data []syscall.NetlinkRouteAttr) error {
	native = nl.NativeEndian()
	fq := qdisc.(*Fq)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_FQ_BUCKETS_LOG:
			fq.Buckets = native.Uint32(datum.Value)
		case nl.TCA_FQ_LOW_RATE_THRESHOLD:
			fq.LowRateThreshold = native.Uint32(datum.Value)
		case nl.TCA_FQ_QUANTUM:
			fq.Quantum = native.Uint32(datum.Value)
		case nl.TCA_FQ_RATE_ENABLE:
			fq.Pacing = native.Uint32(datum.Value)
		case nl.TCA_FQ_INITIAL_QUANTUM:
			fq.InitialQuantum = native.Uint32(datum.Value)
		case nl.TCA_FQ_ORPHAN_MASK:
			// TODO
		case nl.TCA_FQ_FLOW_REFILL_DELAY:
			fq.FlowRefillDelay = native.Uint32(datum.Value)
		case nl.TCA_FQ_FLOW_PLIMIT:
			fq.FlowPacketLimit = native.Uint32(datum.Value)
		case nl.TCA_FQ_PLIMIT:
			fq.PacketLimit = native.Uint32(datum.Value)
		case nl.TCA_FQ_FLOW_MAX_RATE:
			fq.FlowMaxRate = native.Uint32(datum.Value)
		case nl.TCA_FQ_FLOW_DEFAULT_RATE:
			fq.FlowDefaultRate = native.Uint32(datum.Value)
		}
	}
	return nil
}

func parseNetemData(qdisc Qdisc, value []byte) error {
	netem := qdisc.(*Netem)
	opt := nl.DeserializeTcNetemQopt(value)
	netem.Latency = opt.Latency
	netem.Limit = opt.Limit
	netem.Loss = opt.Loss
	netem.Gap = opt.Gap
	netem.Duplicate = opt.Duplicate
	netem.Jitter = opt.Jitter
	data, err := nl.ParseRouteAttr(value[nl.SizeofTcNetemQopt:])
	if err != nil {
		return err
	}
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_NETEM_CORR:
			opt := nl.DeserializeTcNetemCorr(datum.Value)
			netem.DelayCorr = opt.DelayCorr
			netem.LossCorr = opt.LossCorr
			netem.DuplicateCorr = opt.DupCorr
		case nl.TCA_NETEM_CORRUPT:
			opt := nl.DeserializeTcNetemCorrupt(datum.Value)
			netem.CorruptProb = opt.Probability
			netem.CorruptCorr = opt.Correlation
		case nl.TCA_NETEM_REORDER:
			opt := nl.DeserializeTcNetemReorder(datum.Value)
			netem.ReorderProb = opt.Probability
			netem.ReorderCorr = opt.Correlation
		}
	}
	return nil
}

func parseTbfData(qdisc Qdisc, data []syscall.NetlinkRouteAttr) error {
	native = nl.NativeEndian()
	tbf := qdisc.(*Tbf)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_TBF_PARMS:
			opt := nl.DeserializeTcTbfQopt(datum.Value)
			tbf.Rate = uint64(opt.Rate.Rate)
			tbf.Peakrate = uint64(opt.Peakrate.Rate)
			tbf.Limit = opt.Limit
			tbf.Buffer = opt.Buffer
		case nl.TCA_TBF_RATE64:
			tbf.Rate = native.Uint64(datum.Value[0:8])
		case nl.TCA_TBF_PRATE64:
			tbf.Peakrate = native.Uint64(datum.Value[0:8])
		case nl.TCA_TBF_PBURST:
			tbf.Minburst = native.Uint32(datum.Value[0:4])
		}
	}
	return nil
}

const (
	TIME_UNITS_PER_SEC = 1000000
)

var (
	tickInUsec  float64
	clockFactor float64
	hz          float64
)

func initClock() {
	data, err := ioutil.ReadFile("/proc/net/psched")
	if err != nil {
		return
	}
	parts := strings.Split(strings.TrimSpace(string(data)), " ")
	if len(parts) < 3 {
		return
	}
	var vals [3]uint64
	for i := range vals {
		val, err := strconv.ParseUint(parts[i], 16, 32)
		if err != nil {
			return
		}
		vals[i] = val
	}
	// compatibility
	if vals[2] == 1000000000 {
		vals[0] = vals[1]
	}
	clockFactor = float64(vals[2]) / TIME_UNITS_PER_SEC
	tickInUsec = float64(vals[0]) / float64(vals[1]) * clockFactor
	hz = float64(vals[0])
}

func TickInUsec() float64 {
	if tickInUsec == 0.0 {
		initClock()
	}
	return tickInUsec
}

func ClockFactor() float64 {
	if clockFactor == 0.0 {
		initClock()
	}
	return clockFactor
}

func Hz() float64 {
	if hz == 0.0 {
		initClock()
	}
	return hz
}

func time2Tick(time uint32) uint32 {
	return uint32(float64(time) * TickInUsec())
}

func tick2Time(tick uint32) uint32 {
	return uint32(float64(tick) / TickInUsec())
}

func time2Ktime(time uint32) uint32 {
	return uint32(float64(time) * ClockFactor())
}

func ktime2Time(ktime uint32) uint32 {
	return uint32(float64(ktime) / ClockFactor())
}

func burst(rate uint64, buffer uint32) uint32 {
	return uint32(float64(rate) * float64(tick2Time(buffer)) / TIME_UNITS_PER_SEC)
}

func latency(rate uint64, limit, buffer uint32) float64 {
	return TIME_UNITS_PER_SEC*(float64(limit)/float64(rate)) - float64(tick2Time(buffer))
}

func Xmittime(rate uint64, size uint32) float64 {
	return TickInUsec() * TIME_UNITS_PER_SEC * (float64(size) / float64(rate))
}
