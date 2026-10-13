package netlink

import (
	"fmt"
)

// Class interfaces for all classes
type Class interface {
	Attrs() *ClassAttrs
	Type() string
}

// Generic networking statistics for netlink users.
// This file contains "gnet_" prefixed structs and relevant functions.
// See Documentation/networking/getn_stats.txt in Linux source code for more details.

// GnetStatsBasic Ref: struct gnet_stats_basic { ... }
type GnetStatsBasic struct {
	Bytes   uint64 // number of seen bytes
	Packets uint32 // number of seen packets
}

// GnetStatsRateEst Ref: struct gnet_stats_rate_est { ... }
type GnetStatsRateEst struct {
	Bps uint32 // current byte rate
	Pps uint32 // current packet rate
}

// GnetStatsRateEst64 Ref: struct gnet_stats_rate_est64 { ... }
type GnetStatsRateEst64 struct {
	Bps uint64 // current byte rate
	Pps uint64 // current packet rate
}

// GnetStatsQueue Ref: struct gnet_stats_queue { ... }
type GnetStatsQueue struct {
	Qlen       uint32 // queue length
	Backlog    uint32 // backlog size of queue
	Drops      uint32 // number of dropped packets
	Requeues   uint32 // number of requues
	Overlimits uint32 // number of enqueues over the limit
}

// ClassStatistics representation based on generic networking statistics for netlink.
// See Documentation/networking/gen_stats.txt in Linux source code for more details.
type ClassStatistics struct {
	Basic   *GnetStatsBasic
	Queue   *GnetStatsQueue
	RateEst *GnetStatsRateEst
	BasicHw *GnetStatsBasic // Hardward statistics added in kernel 4.20
}

// NewClassStatistics Construct a ClassStatistics struct which fields are all initialized by 0.
func NewClassStatistics() *ClassStatistics {
	return &ClassStatistics{
		Basic:   &GnetStatsBasic{},
		Queue:   &GnetStatsQueue{},
		RateEst: &GnetStatsRateEst{},
		BasicHw: &GnetStatsBasic{},
	}
}

// ClassAttrs represents a netlink class. A filter is associated with a link,
// has a handle and a parent. The root filter of a device should have a
// parent == HANDLE_ROOT.
type ClassAttrs struct {
	LinkIndex  int
	Handle     uint32
	Parent     uint32
	Leaf       uint32
	Statistics *ClassStatistics
}

func (q ClassAttrs) String() string {
	return fmt.Sprintf("{LinkIndex: %d, Handle: %s, Parent: %s, Leaf: %d}", q.LinkIndex, HandleStr(q.Handle), HandleStr(q.Parent), q.Leaf)
}

// HtbClassAttrs stores the attributes of HTB class
type HtbClassAttrs struct {
	// TODO handle all attributes
	Rate    uint64
	Ceil    uint64
	Buffer  uint32
	Cbuffer uint32
	Quantum uint32
	Level   uint32
	Prio    uint32
}

func (q HtbClassAttrs) String() string {
	return fmt.Sprintf("{Rate: %d, Ceil: %d, Buffer: %d, Cbuffer: %d}", q.Rate, q.Ceil, q.Buffer, q.Cbuffer)
}

// HtbClass represents an Htb class
type HtbClass struct {
	ClassAttrs
	Rate    uint64
	Ceil    uint64
	Buffer  uint32
	Cbuffer uint32
	Quantum uint32
	Level   uint32
	Prio    uint32
}

func (q HtbClass) String() string {
	return fmt.Sprintf("{Rate: %d, Ceil: %d, Buffer: %d, Cbuffer: %d}", q.Rate, q.Ceil, q.Buffer, q.Cbuffer)
}

// Attrs returns the class attributes
func (q *HtbClass) Attrs() *ClassAttrs {
	return &q.ClassAttrs
}

// Type return the class type
func (q *HtbClass) Type() string {
	return "htb"
}

// GenericClass classes represent types that are not currently understood
// by this netlink library.
type GenericClass struct {
	ClassAttrs
	ClassType string
}

// Attrs return the class attributes
func (class *GenericClass) Attrs() *ClassAttrs {
	return &class.ClassAttrs
}

// Type return the class type
func (class *GenericClass) Type() string {
	return class.ClassType
}

// ServiceCurve is a nondecreasing function of some time unit, returning the amount of service
// (an allowed or allocated amount of bandwidth) at some specific point in time. The purpose of it
// should be subconsciously obvious: if a class was allowed to transfer not less than the amount
// specified by its service curve, then the service curve is not violated.
type ServiceCurve struct {
	m1 uint32
	d  uint32
	m2 uint32
}

// Attrs return the parameters of the service curve
func (c *ServiceCurve) Attrs() (uint32, uint32, uint32) {
	return c.m1, c.d, c.m2
}

// Burst returns the burst rate (m1) of the curve
func (c *ServiceCurve) Burst() uint32 {
	return c.m1
}

// Delay return the delay (d) of the curve
func (c *ServiceCurve) Delay() uint32 {
	return c.d
}

// Rate returns the rate (m2) of the curve
func (c *ServiceCurve) Rate() uint32 {
	return c.m2
}

// HfscClass is a representation of the HFSC class
type HfscClass struct {
	ClassAttrs
	Rsc ServiceCurve
	Fsc ServiceCurve
	Usc ServiceCurve
}

// SetUsc sets the USC curve. The bandwidth (m1 and m2) is specified in bits and the delay in
// seconds.
func (hfsc *HfscClass) SetUsc(m1 uint32, d uint32, m2 uint32) {
	hfsc.Usc = ServiceCurve{m1: m1, d: d, m2: m2}
}

// SetFsc sets the Fsc curve. The bandwidth (m1 and m2) is specified in bits and the delay in
// seconds.
func (hfsc *HfscClass) SetFsc(m1 uint32, d uint32, m2 uint32) {
	hfsc.Fsc = ServiceCurve{m1: m1, d: d, m2: m2}
}

// SetRsc sets the Rsc curve. The bandwidth (m1 and m2) is specified in bits and the delay in
// seconds.
func (hfsc *HfscClass) SetRsc(m1 uint32, d uint32, m2 uint32) {
	hfsc.Rsc = ServiceCurve{m1: m1, d: d, m2: m2}
}

// SetSC implements the SC from the `tc` CLI. This function behaves the same as if one would set the
// USC through the `tc` command-line tool. This means bandwidth (m1 and m2) is specified in bits and
// the delay in ms.
func (hfsc *HfscClass) SetSC(m1 uint32, d uint32, m2 uint32) {
	hfsc.SetRsc(m1, d, m2)
	hfsc.SetFsc(m1, d, m2)
}

// SetUL implements the UL from the `tc` CLI. This function behaves the same as if one would set the
// USC through the `tc` command-line tool. This means bandwidth (m1 and m2) is specified in bits and
// the delay in ms.
func (hfsc *HfscClass) SetUL(m1 uint32, d uint32, m2 uint32) {
	hfsc.SetUsc(m1, d, m2)
}

// SetLS implements the LS from the `tc` CLI. This function behaves the same as if one would set the
// USC through the `tc` command-line tool. This means bandwidth (m1 and m2) is specified in bits and
// the delay in ms.
func (hfsc *HfscClass) SetLS(m1 uint32, d uint32, m2 uint32) {
	hfsc.SetFsc(m1, d, m2)
}

// NewHfscClass returns a new HFSC struct with the set parameters
func NewHfscClass(attrs ClassAttrs) *HfscClass {
	return &HfscClass{
		ClassAttrs: attrs,
		Rsc:        ServiceCurve{},
		Fsc:        ServiceCurve{},
		Usc:        ServiceCurve{},
	}
}

// String() returns a string that contains the information and attributes of the HFSC class
func (hfsc *HfscClass) String() string {
	return fmt.Sprintf(
		"{%s -- {RSC: {m1=%d d=%d m2=%d}} {FSC: {m1=%d d=%d m2=%d}} {USC: {m1=%d d=%d m2=%d}}}",
		hfsc.Attrs(), hfsc.Rsc.m1*8, hfsc.Rsc.d, hfsc.Rsc.m2*8, hfsc.Fsc.m1*8, hfsc.Fsc.d, hfsc.Fsc.m2*8, hfsc.Usc.m1*8, hfsc.Usc.d, hfsc.Usc.m2*8,
	)
}

// Attrs return the Hfsc parameters
func (hfsc *HfscClass) Attrs() *ClassAttrs {
	return &hfsc.ClassAttrs
}

// Type return the type of the class
func (hfsc *HfscClass) Type() string {
	return "hfsc"
}
