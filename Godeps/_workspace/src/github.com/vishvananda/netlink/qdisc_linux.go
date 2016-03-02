package netlink

import (
	"fmt"
	"io/ioutil"
	"strconv"
	"strings"
	"syscall"

	"github.com/vishvananda/netlink/nl"
)

// QdiscDel will delete a qdisc from the system.
// Equivalent to: `tc qdisc del $qdisc`
func QdiscDel(qdisc Qdisc) error {
	req := nl.NewNetlinkRequest(syscall.RTM_DELQDISC, syscall.NLM_F_ACK)
	base := qdisc.Attrs()
	msg := &nl.TcMsg{
		Family:  nl.FAMILY_ALL,
		Ifindex: int32(base.LinkIndex),
		Handle:  base.Handle,
		Parent:  base.Parent,
	}
	req.AddData(msg)

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// QdiscAdd will add a qdisc to the system.
// Equivalent to: `tc qdisc add $qdisc`
func QdiscAdd(qdisc Qdisc) error {
	req := nl.NewNetlinkRequest(syscall.RTM_NEWQDISC, syscall.NLM_F_CREATE|syscall.NLM_F_EXCL|syscall.NLM_F_ACK)
	base := qdisc.Attrs()
	msg := &nl.TcMsg{
		Family:  nl.FAMILY_ALL,
		Ifindex: int32(base.LinkIndex),
		Handle:  base.Handle,
		Parent:  base.Parent,
	}
	req.AddData(msg)
	req.AddData(nl.NewRtAttr(nl.TCA_KIND, nl.ZeroTerminated(qdisc.Type())))

	options := nl.NewRtAttr(nl.TCA_OPTIONS, nil)
	if prio, ok := qdisc.(*Prio); ok {
		tcmap := nl.TcPrioMap{
			Bands:   int32(prio.Bands),
			Priomap: prio.PriorityMap,
		}
		options = nl.NewRtAttr(nl.TCA_OPTIONS, tcmap.Serialize())
	} else if tbf, ok := qdisc.(*Tbf); ok {
		opt := nl.TcTbfQopt{}
		// TODO: handle rate > uint32
		opt.Rate.Rate = uint32(tbf.Rate)
		opt.Limit = tbf.Limit
		opt.Buffer = tbf.Buffer
		nl.NewRtAttrChild(options, nl.TCA_TBF_PARMS, opt.Serialize())
	} else if _, ok := qdisc.(*Ingress); ok {
		// ingress filters must use the proper handle
		if msg.Parent != HANDLE_INGRESS {
			return fmt.Errorf("Ingress filters must set Parent to HANDLE_INGRESS")
		}
	}
	req.AddData(options)
	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// QdiscList gets a list of qdiscs in the system.
// Equivalent to: `tc qdisc show`.
// The list can be filtered by link.
func QdiscList(link Link) ([]Qdisc, error) {
	req := nl.NewNetlinkRequest(syscall.RTM_GETQDISC, syscall.NLM_F_DUMP)
	index := int32(0)
	if link != nil {
		base := link.Attrs()
		ensureIndex(base)
		index = int32(base.Index)
	}
	msg := &nl.TcMsg{
		Family:  nl.FAMILY_ALL,
		Ifindex: index,
	}
	req.AddData(msg)

	msgs, err := req.Execute(syscall.NETLINK_ROUTE, syscall.RTM_NEWQDISC)
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

func parseTbfData(qdisc Qdisc, data []syscall.NetlinkRouteAttr) error {
	native = nl.NativeEndian()
	tbf := qdisc.(*Tbf)
	for _, datum := range data {
		switch datum.Attr.Type {
		case nl.TCA_TBF_PARMS:
			opt := nl.DeserializeTcTbfQopt(datum.Value)
			tbf.Rate = uint64(opt.Rate.Rate)
			tbf.Limit = opt.Limit
			tbf.Buffer = opt.Buffer
		case nl.TCA_TBF_RATE64:
			tbf.Rate = native.Uint64(datum.Value[0:4])
		}
	}
	return nil
}

const (
	TIME_UNITS_PER_SEC = 1000000
)

var (
	tickInUsec  float64 = 0.0
	clockFactor float64 = 0.0
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
