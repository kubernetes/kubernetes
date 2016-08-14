package netlink

import (
	"fmt"
	"math"
)

const (
	HANDLE_NONE      = 0
	HANDLE_INGRESS   = 0xFFFFFFF1
	HANDLE_CLSACT    = HANDLE_INGRESS
	HANDLE_ROOT      = 0xFFFFFFFF
	PRIORITY_MAP_LEN = 16
)
const (
	HANDLE_MIN_INGRESS = 0xFFFFFFF2
	HANDLE_MIN_EGRESS  = 0xFFFFFFF3
)

type Qdisc interface {
	Attrs() *QdiscAttrs
	Type() string
}

// QdiscAttrs represents a netlink qdisc. A qdisc is associated with a link,
// has a handle, a parent and a refcnt. The root qdisc of a device should
// have parent == HANDLE_ROOT.
type QdiscAttrs struct {
	LinkIndex int
	Handle    uint32
	Parent    uint32
	Refcnt    uint32 // read only
}

func (q QdiscAttrs) String() string {
	return fmt.Sprintf("{LinkIndex: %d, Handle: %s, Parent: %s, Refcnt: %d}", q.LinkIndex, HandleStr(q.Handle), HandleStr(q.Parent), q.Refcnt)
}

func MakeHandle(major, minor uint16) uint32 {
	return (uint32(major) << 16) | uint32(minor)
}

func MajorMinor(handle uint32) (uint16, uint16) {
	return uint16((handle & 0xFFFF0000) >> 16), uint16(handle & 0x0000FFFFF)
}

func HandleStr(handle uint32) string {
	switch handle {
	case HANDLE_NONE:
		return "none"
	case HANDLE_INGRESS:
		return "ingress"
	case HANDLE_ROOT:
		return "root"
	default:
		major, minor := MajorMinor(handle)
		return fmt.Sprintf("%x:%x", major, minor)
	}
}

func Percentage2u32(percentage float32) uint32 {
	// FIXME this is most likely not the best way to convert from % to uint32
	if percentage == 100 {
		return math.MaxUint32
	}
	return uint32(math.MaxUint32 * (percentage / 100))
}

// PfifoFast is the default qdisc created by the kernel if one has not
// been defined for the interface
type PfifoFast struct {
	QdiscAttrs
	Bands       uint8
	PriorityMap [PRIORITY_MAP_LEN]uint8
}

func (qdisc *PfifoFast) Attrs() *QdiscAttrs {
	return &qdisc.QdiscAttrs
}

func (qdisc *PfifoFast) Type() string {
	return "pfifo_fast"
}

// Prio is a basic qdisc that works just like PfifoFast
type Prio struct {
	QdiscAttrs
	Bands       uint8
	PriorityMap [PRIORITY_MAP_LEN]uint8
}

func NewPrio(attrs QdiscAttrs) *Prio {
	return &Prio{
		QdiscAttrs:  attrs,
		Bands:       3,
		PriorityMap: [PRIORITY_MAP_LEN]uint8{1, 2, 2, 2, 1, 2, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1},
	}
}

func (qdisc *Prio) Attrs() *QdiscAttrs {
	return &qdisc.QdiscAttrs
}

func (qdisc *Prio) Type() string {
	return "prio"
}

// Htb is a classful qdisc that rate limits based on tokens
type Htb struct {
	QdiscAttrs
	Version      uint32
	Rate2Quantum uint32
	Defcls       uint32
	Debug        uint32
	DirectPkts   uint32
}

func NewHtb(attrs QdiscAttrs) *Htb {
	return &Htb{
		QdiscAttrs:   attrs,
		Version:      3,
		Defcls:       0,
		Rate2Quantum: 10,
		Debug:        0,
		DirectPkts:   0,
	}
}

func (qdisc *Htb) Attrs() *QdiscAttrs {
	return &qdisc.QdiscAttrs
}

func (qdisc *Htb) Type() string {
	return "htb"
}

// Netem is a classless qdisc that rate limits based on tokens

type NetemQdiscAttrs struct {
	Latency       uint32  // in us
	DelayCorr     float32 // in %
	Limit         uint32
	Loss          float32 // in %
	LossCorr      float32 // in %
	Gap           uint32
	Duplicate     float32 // in %
	DuplicateCorr float32 // in %
	Jitter        uint32  // in us
	ReorderProb   float32 // in %
	ReorderCorr   float32 // in %
	CorruptProb   float32 // in %
	CorruptCorr   float32 // in %
}

func (q NetemQdiscAttrs) String() string {
	return fmt.Sprintf(
		"{Latency: %d, Limit: %d, Loss: %f, Gap: %d, Duplicate: %f, Jitter: %d}",
		q.Latency, q.Limit, q.Loss, q.Gap, q.Duplicate, q.Jitter,
	)
}

type Netem struct {
	QdiscAttrs
	Latency       uint32
	DelayCorr     uint32
	Limit         uint32
	Loss          uint32
	LossCorr      uint32
	Gap           uint32
	Duplicate     uint32
	DuplicateCorr uint32
	Jitter        uint32
	ReorderProb   uint32
	ReorderCorr   uint32
	CorruptProb   uint32
	CorruptCorr   uint32
}

func (qdisc *Netem) Attrs() *QdiscAttrs {
	return &qdisc.QdiscAttrs
}

func (qdisc *Netem) Type() string {
	return "netem"
}

// Tbf is a classless qdisc that rate limits based on tokens
type Tbf struct {
	QdiscAttrs
	// TODO: handle 64bit rate properly
	Rate   uint64
	Limit  uint32
	Buffer uint32
	// TODO: handle other settings
}

func (qdisc *Tbf) Attrs() *QdiscAttrs {
	return &qdisc.QdiscAttrs
}

func (qdisc *Tbf) Type() string {
	return "tbf"
}

// Ingress is a qdisc for adding ingress filters
type Ingress struct {
	QdiscAttrs
}

func (qdisc *Ingress) Attrs() *QdiscAttrs {
	return &qdisc.QdiscAttrs
}

func (qdisc *Ingress) Type() string {
	return "ingress"
}

// GenericQdisc qdiscs represent types that are not currently understood
// by this netlink library.
type GenericQdisc struct {
	QdiscAttrs
	QdiscType string
}

func (qdisc *GenericQdisc) Attrs() *QdiscAttrs {
	return &qdisc.QdiscAttrs
}

func (qdisc *GenericQdisc) Type() string {
	return qdisc.QdiscType
}
