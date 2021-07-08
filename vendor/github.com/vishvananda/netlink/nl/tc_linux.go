package nl

import (
	"encoding/binary"
	"unsafe"
)

// LinkLayer
const (
	LINKLAYER_UNSPEC = iota
	LINKLAYER_ETHERNET
	LINKLAYER_ATM
)

// ATM
const (
	ATM_CELL_PAYLOAD = 48
	ATM_CELL_SIZE    = 53
)

const TC_LINKLAYER_MASK = 0x0F

// Police
const (
	TCA_POLICE_UNSPEC = iota
	TCA_POLICE_TBF
	TCA_POLICE_RATE
	TCA_POLICE_PEAKRATE
	TCA_POLICE_AVRATE
	TCA_POLICE_RESULT
	TCA_POLICE_MAX = TCA_POLICE_RESULT
)

// Message types
const (
	TCA_UNSPEC = iota
	TCA_KIND
	TCA_OPTIONS
	TCA_STATS
	TCA_XSTATS
	TCA_RATE
	TCA_FCNT
	TCA_STATS2
	TCA_STAB
	TCA_MAX = TCA_STAB
)

const (
	TCA_ACT_TAB = 1
	TCAA_MAX    = 1
)

const (
	TCA_ACT_UNSPEC = iota
	TCA_ACT_KIND
	TCA_ACT_OPTIONS
	TCA_ACT_INDEX
	TCA_ACT_STATS
	TCA_ACT_MAX
)

const (
	TCA_PRIO_UNSPEC = iota
	TCA_PRIO_MQ
	TCA_PRIO_MAX = TCA_PRIO_MQ
)

const (
	TCA_STATS_UNSPEC = iota
	TCA_STATS_BASIC
	TCA_STATS_RATE_EST
	TCA_STATS_QUEUE
	TCA_STATS_APP
	TCA_STATS_MAX = TCA_STATS_APP
)

const (
	SizeofTcMsg          = 0x14
	SizeofTcActionMsg    = 0x04
	SizeofTcPrioMap      = 0x14
	SizeofTcRateSpec     = 0x0c
	SizeofTcNetemQopt    = 0x18
	SizeofTcNetemCorr    = 0x0c
	SizeofTcNetemReorder = 0x08
	SizeofTcNetemCorrupt = 0x08
	SizeofTcTbfQopt      = 2*SizeofTcRateSpec + 0x0c
	SizeofTcHtbCopt      = 2*SizeofTcRateSpec + 0x14
	SizeofTcHtbGlob      = 0x14
	SizeofTcU32Key       = 0x10
	SizeofTcU32Sel       = 0x10 // without keys
	SizeofTcGen          = 0x14
	SizeofTcConnmark     = SizeofTcGen + 0x04
	SizeofTcMirred       = SizeofTcGen + 0x08
	SizeofTcTunnelKey    = SizeofTcGen + 0x04
	SizeofTcSkbEdit      = SizeofTcGen
	SizeofTcPolice       = 2*SizeofTcRateSpec + 0x20
	SizeofTcSfqQopt      = 0x0b
	SizeofTcSfqRedStats  = 0x18
	SizeofTcSfqQoptV1    = SizeofTcSfqQopt + SizeofTcSfqRedStats + 0x1c
)

// struct tcmsg {
//   unsigned char tcm_family;
//   unsigned char tcm__pad1;
//   unsigned short  tcm__pad2;
//   int   tcm_ifindex;
//   __u32   tcm_handle;
//   __u32   tcm_parent;
//   __u32   tcm_info;
// };

type TcMsg struct {
	Family  uint8
	Pad     [3]byte
	Ifindex int32
	Handle  uint32
	Parent  uint32
	Info    uint32
}

func (msg *TcMsg) Len() int {
	return SizeofTcMsg
}

func DeserializeTcMsg(b []byte) *TcMsg {
	return (*TcMsg)(unsafe.Pointer(&b[0:SizeofTcMsg][0]))
}

func (x *TcMsg) Serialize() []byte {
	return (*(*[SizeofTcMsg]byte)(unsafe.Pointer(x)))[:]
}

// struct tcamsg {
//   unsigned char tca_family;
//   unsigned char tca__pad1;
//   unsigned short  tca__pad2;
// };

type TcActionMsg struct {
	Family uint8
	Pad    [3]byte
}

func (msg *TcActionMsg) Len() int {
	return SizeofTcActionMsg
}

func DeserializeTcActionMsg(b []byte) *TcActionMsg {
	return (*TcActionMsg)(unsafe.Pointer(&b[0:SizeofTcActionMsg][0]))
}

func (x *TcActionMsg) Serialize() []byte {
	return (*(*[SizeofTcActionMsg]byte)(unsafe.Pointer(x)))[:]
}

const (
	TC_PRIO_MAX = 15
)

// struct tc_prio_qopt {
// 	int bands;      /* Number of bands */
// 	__u8  priomap[TC_PRIO_MAX+1]; /* Map: logical priority -> PRIO band */
// };

type TcPrioMap struct {
	Bands   int32
	Priomap [TC_PRIO_MAX + 1]uint8
}

func (msg *TcPrioMap) Len() int {
	return SizeofTcPrioMap
}

func DeserializeTcPrioMap(b []byte) *TcPrioMap {
	return (*TcPrioMap)(unsafe.Pointer(&b[0:SizeofTcPrioMap][0]))
}

func (x *TcPrioMap) Serialize() []byte {
	return (*(*[SizeofTcPrioMap]byte)(unsafe.Pointer(x)))[:]
}

const (
	TCA_TBF_UNSPEC = iota
	TCA_TBF_PARMS
	TCA_TBF_RTAB
	TCA_TBF_PTAB
	TCA_TBF_RATE64
	TCA_TBF_PRATE64
	TCA_TBF_BURST
	TCA_TBF_PBURST
	TCA_TBF_MAX = TCA_TBF_PBURST
)

// struct tc_ratespec {
//   unsigned char cell_log;
//   __u8    linklayer; /* lower 4 bits */
//   unsigned short  overhead;
//   short   cell_align;
//   unsigned short  mpu;
//   __u32   rate;
// };

type TcRateSpec struct {
	CellLog   uint8
	Linklayer uint8
	Overhead  uint16
	CellAlign int16
	Mpu       uint16
	Rate      uint32
}

func (msg *TcRateSpec) Len() int {
	return SizeofTcRateSpec
}

func DeserializeTcRateSpec(b []byte) *TcRateSpec {
	return (*TcRateSpec)(unsafe.Pointer(&b[0:SizeofTcRateSpec][0]))
}

func (x *TcRateSpec) Serialize() []byte {
	return (*(*[SizeofTcRateSpec]byte)(unsafe.Pointer(x)))[:]
}

/**
* NETEM
 */

const (
	TCA_NETEM_UNSPEC = iota
	TCA_NETEM_CORR
	TCA_NETEM_DELAY_DIST
	TCA_NETEM_REORDER
	TCA_NETEM_CORRUPT
	TCA_NETEM_LOSS
	TCA_NETEM_RATE
	TCA_NETEM_ECN
	TCA_NETEM_RATE64
	TCA_NETEM_MAX = TCA_NETEM_RATE64
)

// struct tc_netem_qopt {
//	__u32	latency;	/* added delay (us) */
//	__u32   limit;		/* fifo limit (packets) */
//	__u32	loss;		/* random packet loss (0=none ~0=100%) */
//	__u32	gap;		/* re-ordering gap (0 for none) */
//	__u32   duplicate;	/* random packet dup  (0=none ~0=100%) */
// 	__u32	jitter;		/* random jitter in latency (us) */
// };

type TcNetemQopt struct {
	Latency   uint32
	Limit     uint32
	Loss      uint32
	Gap       uint32
	Duplicate uint32
	Jitter    uint32
}

func (msg *TcNetemQopt) Len() int {
	return SizeofTcNetemQopt
}

func DeserializeTcNetemQopt(b []byte) *TcNetemQopt {
	return (*TcNetemQopt)(unsafe.Pointer(&b[0:SizeofTcNetemQopt][0]))
}

func (x *TcNetemQopt) Serialize() []byte {
	return (*(*[SizeofTcNetemQopt]byte)(unsafe.Pointer(x)))[:]
}

// struct tc_netem_corr {
//  __u32   delay_corr; /* delay correlation */
//  __u32   loss_corr;  /* packet loss correlation */
//  __u32   dup_corr;   /* duplicate correlation  */
// };

type TcNetemCorr struct {
	DelayCorr uint32
	LossCorr  uint32
	DupCorr   uint32
}

func (msg *TcNetemCorr) Len() int {
	return SizeofTcNetemCorr
}

func DeserializeTcNetemCorr(b []byte) *TcNetemCorr {
	return (*TcNetemCorr)(unsafe.Pointer(&b[0:SizeofTcNetemCorr][0]))
}

func (x *TcNetemCorr) Serialize() []byte {
	return (*(*[SizeofTcNetemCorr]byte)(unsafe.Pointer(x)))[:]
}

// struct tc_netem_reorder {
//  __u32   probability;
//  __u32   correlation;
// };

type TcNetemReorder struct {
	Probability uint32
	Correlation uint32
}

func (msg *TcNetemReorder) Len() int {
	return SizeofTcNetemReorder
}

func DeserializeTcNetemReorder(b []byte) *TcNetemReorder {
	return (*TcNetemReorder)(unsafe.Pointer(&b[0:SizeofTcNetemReorder][0]))
}

func (x *TcNetemReorder) Serialize() []byte {
	return (*(*[SizeofTcNetemReorder]byte)(unsafe.Pointer(x)))[:]
}

// struct tc_netem_corrupt {
//  __u32   probability;
//  __u32   correlation;
// };

type TcNetemCorrupt struct {
	Probability uint32
	Correlation uint32
}

func (msg *TcNetemCorrupt) Len() int {
	return SizeofTcNetemCorrupt
}

func DeserializeTcNetemCorrupt(b []byte) *TcNetemCorrupt {
	return (*TcNetemCorrupt)(unsafe.Pointer(&b[0:SizeofTcNetemCorrupt][0]))
}

func (x *TcNetemCorrupt) Serialize() []byte {
	return (*(*[SizeofTcNetemCorrupt]byte)(unsafe.Pointer(x)))[:]
}

// struct tc_tbf_qopt {
//   struct tc_ratespec rate;
//   struct tc_ratespec peakrate;
//   __u32   limit;
//   __u32   buffer;
//   __u32   mtu;
// };

type TcTbfQopt struct {
	Rate     TcRateSpec
	Peakrate TcRateSpec
	Limit    uint32
	Buffer   uint32
	Mtu      uint32
}

func (msg *TcTbfQopt) Len() int {
	return SizeofTcTbfQopt
}

func DeserializeTcTbfQopt(b []byte) *TcTbfQopt {
	return (*TcTbfQopt)(unsafe.Pointer(&b[0:SizeofTcTbfQopt][0]))
}

func (x *TcTbfQopt) Serialize() []byte {
	return (*(*[SizeofTcTbfQopt]byte)(unsafe.Pointer(x)))[:]
}

const (
	TCA_HTB_UNSPEC = iota
	TCA_HTB_PARMS
	TCA_HTB_INIT
	TCA_HTB_CTAB
	TCA_HTB_RTAB
	TCA_HTB_DIRECT_QLEN
	TCA_HTB_RATE64
	TCA_HTB_CEIL64
	TCA_HTB_MAX = TCA_HTB_CEIL64
)

//struct tc_htb_opt {
//	struct tc_ratespec	rate;
//	struct tc_ratespec	ceil;
//	__u32	buffer;
//	__u32	cbuffer;
//	__u32	quantum;
//	__u32	level;		/* out only */
//	__u32	prio;
//};

type TcHtbCopt struct {
	Rate    TcRateSpec
	Ceil    TcRateSpec
	Buffer  uint32
	Cbuffer uint32
	Quantum uint32
	Level   uint32
	Prio    uint32
}

func (msg *TcHtbCopt) Len() int {
	return SizeofTcHtbCopt
}

func DeserializeTcHtbCopt(b []byte) *TcHtbCopt {
	return (*TcHtbCopt)(unsafe.Pointer(&b[0:SizeofTcHtbCopt][0]))
}

func (x *TcHtbCopt) Serialize() []byte {
	return (*(*[SizeofTcHtbCopt]byte)(unsafe.Pointer(x)))[:]
}

type TcHtbGlob struct {
	Version      uint32
	Rate2Quantum uint32
	Defcls       uint32
	Debug        uint32
	DirectPkts   uint32
}

func (msg *TcHtbGlob) Len() int {
	return SizeofTcHtbGlob
}

func DeserializeTcHtbGlob(b []byte) *TcHtbGlob {
	return (*TcHtbGlob)(unsafe.Pointer(&b[0:SizeofTcHtbGlob][0]))
}

func (x *TcHtbGlob) Serialize() []byte {
	return (*(*[SizeofTcHtbGlob]byte)(unsafe.Pointer(x)))[:]
}

// HFSC

type Curve struct {
	m1 uint32
	d  uint32
	m2 uint32
}

type HfscCopt struct {
	Rsc Curve
	Fsc Curve
	Usc Curve
}

func (c *Curve) Attrs() (uint32, uint32, uint32) {
	return c.m1, c.d, c.m2
}

func (c *Curve) Set(m1 uint32, d uint32, m2 uint32) {
	c.m1 = m1
	c.d = d
	c.m2 = m2
}

func DeserializeHfscCurve(b []byte) *Curve {
	return &Curve{
		m1: binary.LittleEndian.Uint32(b[0:4]),
		d:  binary.LittleEndian.Uint32(b[4:8]),
		m2: binary.LittleEndian.Uint32(b[8:12]),
	}
}

func SerializeHfscCurve(c *Curve) (b []byte) {
	t := make([]byte, binary.MaxVarintLen32)
	binary.LittleEndian.PutUint32(t, c.m1)
	b = append(b, t[:4]...)
	binary.LittleEndian.PutUint32(t, c.d)
	b = append(b, t[:4]...)
	binary.LittleEndian.PutUint32(t, c.m2)
	b = append(b, t[:4]...)
	return b
}

type TcHfscOpt struct {
	Defcls uint16
}

func (x *TcHfscOpt) Serialize() []byte {
	return (*(*[2]byte)(unsafe.Pointer(x)))[:]
}

const (
	TCA_U32_UNSPEC = iota
	TCA_U32_CLASSID
	TCA_U32_HASH
	TCA_U32_LINK
	TCA_U32_DIVISOR
	TCA_U32_SEL
	TCA_U32_POLICE
	TCA_U32_ACT
	TCA_U32_INDEV
	TCA_U32_PCNT
	TCA_U32_MARK
	TCA_U32_MAX = TCA_U32_MARK
)

// struct tc_u32_key {
//   __be32    mask;
//   __be32    val;
//   int   off;
//   int   offmask;
// };

type TcU32Key struct {
	Mask    uint32 // big endian
	Val     uint32 // big endian
	Off     int32
	OffMask int32
}

func (msg *TcU32Key) Len() int {
	return SizeofTcU32Key
}

func DeserializeTcU32Key(b []byte) *TcU32Key {
	return (*TcU32Key)(unsafe.Pointer(&b[0:SizeofTcU32Key][0]))
}

func (x *TcU32Key) Serialize() []byte {
	return (*(*[SizeofTcU32Key]byte)(unsafe.Pointer(x)))[:]
}

// struct tc_u32_sel {
//   unsigned char   flags;
//   unsigned char   offshift;
//   unsigned char   nkeys;
//
//   __be16      offmask;
//   __u16     off;
//   short     offoff;
//
//   short     hoff;
//   __be32      hmask;
//   struct tc_u32_key keys[0];
// };

const (
	TC_U32_TERMINAL  = 1 << iota
	TC_U32_OFFSET    = 1 << iota
	TC_U32_VAROFFSET = 1 << iota
	TC_U32_EAT       = 1 << iota
)

type TcU32Sel struct {
	Flags    uint8
	Offshift uint8
	Nkeys    uint8
	Pad      uint8
	Offmask  uint16 // big endian
	Off      uint16
	Offoff   int16
	Hoff     int16
	Hmask    uint32 // big endian
	Keys     []TcU32Key
}

func (msg *TcU32Sel) Len() int {
	return SizeofTcU32Sel + int(msg.Nkeys)*SizeofTcU32Key
}

func DeserializeTcU32Sel(b []byte) *TcU32Sel {
	x := &TcU32Sel{}
	copy((*(*[SizeofTcU32Sel]byte)(unsafe.Pointer(x)))[:], b)
	next := SizeofTcU32Sel
	var i uint8
	for i = 0; i < x.Nkeys; i++ {
		x.Keys = append(x.Keys, *DeserializeTcU32Key(b[next:]))
		next += SizeofTcU32Key
	}
	return x
}

func (x *TcU32Sel) Serialize() []byte {
	// This can't just unsafe.cast because it must iterate through keys.
	buf := make([]byte, x.Len())
	copy(buf, (*(*[SizeofTcU32Sel]byte)(unsafe.Pointer(x)))[:])
	next := SizeofTcU32Sel
	for _, key := range x.Keys {
		keyBuf := key.Serialize()
		copy(buf[next:], keyBuf)
		next += SizeofTcU32Key
	}
	return buf
}

type TcGen struct {
	Index   uint32
	Capab   uint32
	Action  int32
	Refcnt  int32
	Bindcnt int32
}

func (msg *TcGen) Len() int {
	return SizeofTcGen
}

func DeserializeTcGen(b []byte) *TcGen {
	return (*TcGen)(unsafe.Pointer(&b[0:SizeofTcGen][0]))
}

func (x *TcGen) Serialize() []byte {
	return (*(*[SizeofTcGen]byte)(unsafe.Pointer(x)))[:]
}

// #define tc_gen \
//   __u32                 index; \
//   __u32                 capab; \
//   int                   action; \
//   int                   refcnt; \
//   int                   bindcnt

const (
	TCA_ACT_GACT = 5
)

const (
	TCA_GACT_UNSPEC = iota
	TCA_GACT_TM
	TCA_GACT_PARMS
	TCA_GACT_PROB
	TCA_GACT_MAX = TCA_GACT_PROB
)

type TcGact TcGen

const (
	TCA_ACT_BPF = 13
)

const (
	TCA_ACT_BPF_UNSPEC = iota
	TCA_ACT_BPF_TM
	TCA_ACT_BPF_PARMS
	TCA_ACT_BPF_OPS_LEN
	TCA_ACT_BPF_OPS
	TCA_ACT_BPF_FD
	TCA_ACT_BPF_NAME
	TCA_ACT_BPF_MAX = TCA_ACT_BPF_NAME
)

const (
	TCA_BPF_FLAG_ACT_DIRECT uint32 = 1 << iota
)

const (
	TCA_BPF_UNSPEC = iota
	TCA_BPF_ACT
	TCA_BPF_POLICE
	TCA_BPF_CLASSID
	TCA_BPF_OPS_LEN
	TCA_BPF_OPS
	TCA_BPF_FD
	TCA_BPF_NAME
	TCA_BPF_FLAGS
	TCA_BPF_FLAGS_GEN
	TCA_BPF_TAG
	TCA_BPF_ID
	TCA_BPF_MAX = TCA_BPF_ID
)

type TcBpf TcGen

const (
	TCA_ACT_CONNMARK = 14
)

const (
	TCA_CONNMARK_UNSPEC = iota
	TCA_CONNMARK_PARMS
	TCA_CONNMARK_TM
	TCA_CONNMARK_MAX = TCA_CONNMARK_TM
)

// struct tc_connmark {
//   tc_gen;
//   __u16 zone;
// };

type TcConnmark struct {
	TcGen
	Zone uint16
}

func (msg *TcConnmark) Len() int {
	return SizeofTcConnmark
}

func DeserializeTcConnmark(b []byte) *TcConnmark {
	return (*TcConnmark)(unsafe.Pointer(&b[0:SizeofTcConnmark][0]))
}

func (x *TcConnmark) Serialize() []byte {
	return (*(*[SizeofTcConnmark]byte)(unsafe.Pointer(x)))[:]
}

const (
	TCA_ACT_MIRRED = 8
)

const (
	TCA_MIRRED_UNSPEC = iota
	TCA_MIRRED_TM
	TCA_MIRRED_PARMS
	TCA_MIRRED_MAX = TCA_MIRRED_PARMS
)

// struct tc_mirred {
// 	tc_gen;
// 	int                     eaction;   /* one of IN/EGRESS_MIRROR/REDIR */
// 	__u32                   ifindex;  /* ifindex of egress port */
// };

type TcMirred struct {
	TcGen
	Eaction int32
	Ifindex uint32
}

func (msg *TcMirred) Len() int {
	return SizeofTcMirred
}

func DeserializeTcMirred(b []byte) *TcMirred {
	return (*TcMirred)(unsafe.Pointer(&b[0:SizeofTcMirred][0]))
}

func (x *TcMirred) Serialize() []byte {
	return (*(*[SizeofTcMirred]byte)(unsafe.Pointer(x)))[:]
}

const (
	TCA_TUNNEL_KEY_UNSPEC = iota
	TCA_TUNNEL_KEY_TM
	TCA_TUNNEL_KEY_PARMS
	TCA_TUNNEL_KEY_ENC_IPV4_SRC
	TCA_TUNNEL_KEY_ENC_IPV4_DST
	TCA_TUNNEL_KEY_ENC_IPV6_SRC
	TCA_TUNNEL_KEY_ENC_IPV6_DST
	TCA_TUNNEL_KEY_ENC_KEY_ID
	TCA_TUNNEL_KEY_PAD
	TCA_TUNNEL_KEY_ENC_DST_PORT
	TCA_TUNNEL_KEY_NO_CSUM
	TCA_TUNNEL_KEY_ENC_OPTS
	TCA_TUNNEL_KEY_ENC_TOS
	TCA_TUNNEL_KEY_ENC_TTL
	TCA_TUNNEL_KEY_MAX
)

type TcTunnelKey struct {
	TcGen
	Action int32
}

func (x *TcTunnelKey) Len() int {
	return SizeofTcTunnelKey
}

func DeserializeTunnelKey(b []byte) *TcTunnelKey {
	return (*TcTunnelKey)(unsafe.Pointer(&b[0:SizeofTcTunnelKey][0]))
}

func (x *TcTunnelKey) Serialize() []byte {
	return (*(*[SizeofTcTunnelKey]byte)(unsafe.Pointer(x)))[:]
}

const (
	TCA_SKBEDIT_UNSPEC = iota
	TCA_SKBEDIT_TM
	TCA_SKBEDIT_PARMS
	TCA_SKBEDIT_PRIORITY
	TCA_SKBEDIT_QUEUE_MAPPING
	TCA_SKBEDIT_MARK
	TCA_SKBEDIT_PAD
	TCA_SKBEDIT_PTYPE
	TCA_SKBEDIT_MAX = TCA_SKBEDIT_MARK
)

type TcSkbEdit struct {
	TcGen
}

func (x *TcSkbEdit) Len() int {
	return SizeofTcSkbEdit
}

func DeserializeSkbEdit(b []byte) *TcSkbEdit {
	return (*TcSkbEdit)(unsafe.Pointer(&b[0:SizeofTcSkbEdit][0]))
}

func (x *TcSkbEdit) Serialize() []byte {
	return (*(*[SizeofTcSkbEdit]byte)(unsafe.Pointer(x)))[:]
}

// struct tc_police {
// 	__u32			index;
// 	int			action;
// 	__u32			limit;
// 	__u32			burst;
// 	__u32			mtu;
// 	struct tc_ratespec	rate;
// 	struct tc_ratespec	peakrate;
// 	int				refcnt;
// 	int				bindcnt;
// 	__u32			capab;
// };

type TcPolice struct {
	Index    uint32
	Action   int32
	Limit    uint32
	Burst    uint32
	Mtu      uint32
	Rate     TcRateSpec
	PeakRate TcRateSpec
	Refcnt   int32
	Bindcnt  int32
	Capab    uint32
}

func (msg *TcPolice) Len() int {
	return SizeofTcPolice
}

func DeserializeTcPolice(b []byte) *TcPolice {
	return (*TcPolice)(unsafe.Pointer(&b[0:SizeofTcPolice][0]))
}

func (x *TcPolice) Serialize() []byte {
	return (*(*[SizeofTcPolice]byte)(unsafe.Pointer(x)))[:]
}

const (
	TCA_FW_UNSPEC = iota
	TCA_FW_CLASSID
	TCA_FW_POLICE
	TCA_FW_INDEV
	TCA_FW_ACT
	TCA_FW_MASK
	TCA_FW_MAX = TCA_FW_MASK
)

const (
	TCA_MATCHALL_UNSPEC = iota
	TCA_MATCHALL_CLASSID
	TCA_MATCHALL_ACT
	TCA_MATCHALL_FLAGS
)

const (
	TCA_FQ_UNSPEC             = iota
	TCA_FQ_PLIMIT             // limit of total number of packets in queue
	TCA_FQ_FLOW_PLIMIT        // limit of packets per flow
	TCA_FQ_QUANTUM            // RR quantum
	TCA_FQ_INITIAL_QUANTUM    // RR quantum for new flow
	TCA_FQ_RATE_ENABLE        // enable/disable rate limiting
	TCA_FQ_FLOW_DEFAULT_RATE  // obsolete do not use
	TCA_FQ_FLOW_MAX_RATE      // per flow max rate
	TCA_FQ_BUCKETS_LOG        // log2(number of buckets)
	TCA_FQ_FLOW_REFILL_DELAY  // flow credit refill delay in usec
	TCA_FQ_ORPHAN_MASK        // mask applied to orphaned skb hashes
	TCA_FQ_LOW_RATE_THRESHOLD // per packet delay under this rate
)

const (
	TCA_FQ_CODEL_UNSPEC = iota
	TCA_FQ_CODEL_TARGET
	TCA_FQ_CODEL_LIMIT
	TCA_FQ_CODEL_INTERVAL
	TCA_FQ_CODEL_ECN
	TCA_FQ_CODEL_FLOWS
	TCA_FQ_CODEL_QUANTUM
	TCA_FQ_CODEL_CE_THRESHOLD
	TCA_FQ_CODEL_DROP_BATCH_SIZE
	TCA_FQ_CODEL_MEMORY_LIMIT
)

const (
	TCA_HFSC_UNSPEC = iota
	TCA_HFSC_RSC
	TCA_HFSC_FSC
	TCA_HFSC_USC
)

// struct tc_sfq_qopt {
// 	unsigned	quantum;	/* Bytes per round allocated to flow */
// 	int		perturb_period;	/* Period of hash perturbation */
// 	__u32		limit;		/* Maximal packets in queue */
// 	unsigned	divisor;	/* Hash divisor  */
// 	unsigned	flows;		/* Maximal number of flows  */
// };

type TcSfqQopt struct {
	Quantum uint8
	Perturb int32
	Limit   uint32
	Divisor uint8
	Flows   uint8
}

func (x *TcSfqQopt) Len() int {
	return SizeofTcSfqQopt
}

func DeserializeTcSfqQopt(b []byte) *TcSfqQopt {
	return (*TcSfqQopt)(unsafe.Pointer(&b[0:SizeofTcSfqQopt][0]))
}

func (x *TcSfqQopt) Serialize() []byte {
	return (*(*[SizeofTcSfqQopt]byte)(unsafe.Pointer(x)))[:]
}

// struct tc_sfqred_stats {
// 	__u32           prob_drop;      /* Early drops, below max threshold */
// 	__u32           forced_drop;	/* Early drops, after max threshold */
// 	__u32           prob_mark;      /* Marked packets, below max threshold */
// 	__u32           forced_mark;    /* Marked packets, after max threshold */
// 	__u32           prob_mark_head; /* Marked packets, below max threshold */
// 	__u32           forced_mark_head;/* Marked packets, after max threshold */
// };
type TcSfqRedStats struct {
	ProbDrop       uint32
	ForcedDrop     uint32
	ProbMark       uint32
	ForcedMark     uint32
	ProbMarkHead   uint32
	ForcedMarkHead uint32
}

func (x *TcSfqRedStats) Len() int {
	return SizeofTcSfqRedStats
}

func DeserializeTcSfqRedStats(b []byte) *TcSfqRedStats {
	return (*TcSfqRedStats)(unsafe.Pointer(&b[0:SizeofTcSfqRedStats][0]))
}

func (x *TcSfqRedStats) Serialize() []byte {
	return (*(*[SizeofTcSfqRedStats]byte)(unsafe.Pointer(x)))[:]
}

// struct tc_sfq_qopt_v1 {
// 	struct tc_sfq_qopt v0;
// 	unsigned int	depth;		/* max number of packets per flow */
// 	unsigned int	headdrop;
// /* SFQRED parameters */
// 	__u32		limit;		/* HARD maximal flow queue length (bytes) */
// 	__u32		qth_min;	/* Min average length threshold (bytes) */
// 	__u32		qth_max;	/* Max average length threshold (bytes) */
// 	unsigned char   Wlog;		/* log(W)		*/
// 	unsigned char   Plog;		/* log(P_max/(qth_max-qth_min))	*/
// 	unsigned char   Scell_log;	/* cell size for idle damping */
// 	unsigned char	flags;
// 	__u32		max_P;		/* probability, high resolution */
// /* SFQRED stats */
// 	struct tc_sfqred_stats stats;
// };
type TcSfqQoptV1 struct {
	TcSfqQopt
	Depth    uint32
	HeadDrop uint32
	Limit    uint32
	QthMin   uint32
	QthMax   uint32
	Wlog     byte
	Plog     byte
	ScellLog byte
	Flags    byte
	MaxP     uint32
	TcSfqRedStats
}

func (x *TcSfqQoptV1) Len() int {
	return SizeofTcSfqQoptV1
}

func DeserializeTcSfqQoptV1(b []byte) *TcSfqQoptV1 {
	return (*TcSfqQoptV1)(unsafe.Pointer(&b[0:SizeofTcSfqQoptV1][0]))
}

func (x *TcSfqQoptV1) Serialize() []byte {
	return (*(*[SizeofTcSfqQoptV1]byte)(unsafe.Pointer(x)))[:]
}
