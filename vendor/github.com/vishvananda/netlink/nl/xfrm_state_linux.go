package nl

import (
	"unsafe"
)

const (
	SizeofXfrmUsersaId       = 0x18
	SizeofXfrmStats          = 0x0c
	SizeofXfrmUsersaInfo     = 0xe0
	SizeofXfrmUserSpiInfo    = 0xe8
	SizeofXfrmAlgo           = 0x44
	SizeofXfrmAlgoAuth       = 0x48
	SizeofXfrmAlgoAEAD       = 0x48
	SizeofXfrmEncapTmpl      = 0x18
	SizeofXfrmUsersaFlush    = 0x8
	SizeofXfrmReplayStateEsn = 0x18
)

const (
	XFRM_STATE_NOECN      = 1
	XFRM_STATE_DECAP_DSCP = 2
	XFRM_STATE_NOPMTUDISC = 4
	XFRM_STATE_WILDRECV   = 8
	XFRM_STATE_ICMP       = 16
	XFRM_STATE_AF_UNSPEC  = 32
	XFRM_STATE_ALIGN4     = 64
	XFRM_STATE_ESN        = 128
)

// struct xfrm_usersa_id {
//   xfrm_address_t      daddr;
//   __be32        spi;
//   __u16       family;
//   __u8        proto;
// };

type XfrmUsersaId struct {
	Daddr  XfrmAddress
	Spi    uint32 // big endian
	Family uint16
	Proto  uint8
	Pad    byte
}

func (msg *XfrmUsersaId) Len() int {
	return SizeofXfrmUsersaId
}

func DeserializeXfrmUsersaId(b []byte) *XfrmUsersaId {
	return (*XfrmUsersaId)(unsafe.Pointer(&b[0:SizeofXfrmUsersaId][0]))
}

func (msg *XfrmUsersaId) Serialize() []byte {
	return (*(*[SizeofXfrmUsersaId]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_stats {
//   __u32 replay_window;
//   __u32 replay;
//   __u32 integrity_failed;
// };

type XfrmStats struct {
	ReplayWindow    uint32
	Replay          uint32
	IntegrityFailed uint32
}

func (msg *XfrmStats) Len() int {
	return SizeofXfrmStats
}

func DeserializeXfrmStats(b []byte) *XfrmStats {
	return (*XfrmStats)(unsafe.Pointer(&b[0:SizeofXfrmStats][0]))
}

func (msg *XfrmStats) Serialize() []byte {
	return (*(*[SizeofXfrmStats]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_usersa_info {
//   struct xfrm_selector    sel;
//   struct xfrm_id      id;
//   xfrm_address_t      saddr;
//   struct xfrm_lifetime_cfg  lft;
//   struct xfrm_lifetime_cur  curlft;
//   struct xfrm_stats   stats;
//   __u32       seq;
//   __u32       reqid;
//   __u16       family;
//   __u8        mode;   /* XFRM_MODE_xxx */
//   __u8        replay_window;
//   __u8        flags;
// #define XFRM_STATE_NOECN  1
// #define XFRM_STATE_DECAP_DSCP 2
// #define XFRM_STATE_NOPMTUDISC 4
// #define XFRM_STATE_WILDRECV 8
// #define XFRM_STATE_ICMP   16
// #define XFRM_STATE_AF_UNSPEC  32
// #define XFRM_STATE_ALIGN4 64
// #define XFRM_STATE_ESN    128
// };
//
// #define XFRM_SA_XFLAG_DONT_ENCAP_DSCP 1
//

type XfrmUsersaInfo struct {
	Sel          XfrmSelector
	Id           XfrmId
	Saddr        XfrmAddress
	Lft          XfrmLifetimeCfg
	Curlft       XfrmLifetimeCur
	Stats        XfrmStats
	Seq          uint32
	Reqid        uint32
	Family       uint16
	Mode         uint8
	ReplayWindow uint8
	Flags        uint8
	Pad          [7]byte
}

func (msg *XfrmUsersaInfo) Len() int {
	return SizeofXfrmUsersaInfo
}

func DeserializeXfrmUsersaInfo(b []byte) *XfrmUsersaInfo {
	return (*XfrmUsersaInfo)(unsafe.Pointer(&b[0:SizeofXfrmUsersaInfo][0]))
}

func (msg *XfrmUsersaInfo) Serialize() []byte {
	return (*(*[SizeofXfrmUsersaInfo]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_userspi_info {
// 	struct xfrm_usersa_info		info;
// 	__u32				min;
// 	__u32				max;
// };

type XfrmUserSpiInfo struct {
	XfrmUsersaInfo XfrmUsersaInfo
	Min            uint32
	Max            uint32
}

func (msg *XfrmUserSpiInfo) Len() int {
	return SizeofXfrmUserSpiInfo
}

func DeserializeXfrmUserSpiInfo(b []byte) *XfrmUserSpiInfo {
	return (*XfrmUserSpiInfo)(unsafe.Pointer(&b[0:SizeofXfrmUserSpiInfo][0]))
}

func (msg *XfrmUserSpiInfo) Serialize() []byte {
	return (*(*[SizeofXfrmUserSpiInfo]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_algo {
//   char    alg_name[64];
//   unsigned int  alg_key_len;    /* in bits */
//   char    alg_key[0];
// };

type XfrmAlgo struct {
	AlgName   [64]byte
	AlgKeyLen uint32
	AlgKey    []byte
}

func (msg *XfrmAlgo) Len() int {
	return SizeofXfrmAlgo + int(msg.AlgKeyLen/8)
}

func DeserializeXfrmAlgo(b []byte) *XfrmAlgo {
	ret := XfrmAlgo{}
	copy(ret.AlgName[:], b[0:64])
	ret.AlgKeyLen = *(*uint32)(unsafe.Pointer(&b[64]))
	ret.AlgKey = b[68:ret.Len()]
	return &ret
}

func (msg *XfrmAlgo) Serialize() []byte {
	b := make([]byte, msg.Len())
	copy(b[0:64], msg.AlgName[:])
	copy(b[64:68], (*(*[4]byte)(unsafe.Pointer(&msg.AlgKeyLen)))[:])
	copy(b[68:msg.Len()], msg.AlgKey[:])
	return b
}

// struct xfrm_algo_auth {
//   char    alg_name[64];
//   unsigned int  alg_key_len;    /* in bits */
//   unsigned int  alg_trunc_len;  /* in bits */
//   char    alg_key[0];
// };

type XfrmAlgoAuth struct {
	AlgName     [64]byte
	AlgKeyLen   uint32
	AlgTruncLen uint32
	AlgKey      []byte
}

func (msg *XfrmAlgoAuth) Len() int {
	return SizeofXfrmAlgoAuth + int(msg.AlgKeyLen/8)
}

func DeserializeXfrmAlgoAuth(b []byte) *XfrmAlgoAuth {
	ret := XfrmAlgoAuth{}
	copy(ret.AlgName[:], b[0:64])
	ret.AlgKeyLen = *(*uint32)(unsafe.Pointer(&b[64]))
	ret.AlgTruncLen = *(*uint32)(unsafe.Pointer(&b[68]))
	ret.AlgKey = b[72:ret.Len()]
	return &ret
}

func (msg *XfrmAlgoAuth) Serialize() []byte {
	b := make([]byte, msg.Len())
	copy(b[0:64], msg.AlgName[:])
	copy(b[64:68], (*(*[4]byte)(unsafe.Pointer(&msg.AlgKeyLen)))[:])
	copy(b[68:72], (*(*[4]byte)(unsafe.Pointer(&msg.AlgTruncLen)))[:])
	copy(b[72:msg.Len()], msg.AlgKey[:])
	return b
}

// struct xfrm_algo_aead {
//   char    alg_name[64];
//   unsigned int  alg_key_len;  /* in bits */
//   unsigned int  alg_icv_len;  /* in bits */
//   char    alg_key[0];
// }

type XfrmAlgoAEAD struct {
	AlgName   [64]byte
	AlgKeyLen uint32
	AlgICVLen uint32
	AlgKey    []byte
}

func (msg *XfrmAlgoAEAD) Len() int {
	return SizeofXfrmAlgoAEAD + int(msg.AlgKeyLen/8)
}

func DeserializeXfrmAlgoAEAD(b []byte) *XfrmAlgoAEAD {
	ret := XfrmAlgoAEAD{}
	copy(ret.AlgName[:], b[0:64])
	ret.AlgKeyLen = *(*uint32)(unsafe.Pointer(&b[64]))
	ret.AlgICVLen = *(*uint32)(unsafe.Pointer(&b[68]))
	ret.AlgKey = b[72:ret.Len()]
	return &ret
}

func (msg *XfrmAlgoAEAD) Serialize() []byte {
	b := make([]byte, msg.Len())
	copy(b[0:64], msg.AlgName[:])
	copy(b[64:68], (*(*[4]byte)(unsafe.Pointer(&msg.AlgKeyLen)))[:])
	copy(b[68:72], (*(*[4]byte)(unsafe.Pointer(&msg.AlgICVLen)))[:])
	copy(b[72:msg.Len()], msg.AlgKey[:])
	return b
}

// struct xfrm_encap_tmpl {
//   __u16   encap_type;
//   __be16    encap_sport;
//   __be16    encap_dport;
//   xfrm_address_t  encap_oa;
// };

type XfrmEncapTmpl struct {
	EncapType  uint16
	EncapSport uint16 // big endian
	EncapDport uint16 // big endian
	Pad        [2]byte
	EncapOa    XfrmAddress
}

func (msg *XfrmEncapTmpl) Len() int {
	return SizeofXfrmEncapTmpl
}

func DeserializeXfrmEncapTmpl(b []byte) *XfrmEncapTmpl {
	return (*XfrmEncapTmpl)(unsafe.Pointer(&b[0:SizeofXfrmEncapTmpl][0]))
}

func (msg *XfrmEncapTmpl) Serialize() []byte {
	return (*(*[SizeofXfrmEncapTmpl]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_usersa_flush {
//    __u8 proto;
// };

type XfrmUsersaFlush struct {
	Proto uint8
}

func (msg *XfrmUsersaFlush) Len() int {
	return SizeofXfrmUsersaFlush
}

func DeserializeXfrmUsersaFlush(b []byte) *XfrmUsersaFlush {
	return (*XfrmUsersaFlush)(unsafe.Pointer(&b[0:SizeofXfrmUsersaFlush][0]))
}

func (msg *XfrmUsersaFlush) Serialize() []byte {
	return (*(*[SizeofXfrmUsersaFlush]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_replay_state_esn {
//     unsigned int    bmp_len;
//     __u32           oseq;
//     __u32           seq;
//     __u32           oseq_hi;
//     __u32           seq_hi;
//     __u32           replay_window;
//     __u32           bmp[0];
// };

type XfrmReplayStateEsn struct {
	BmpLen       uint32
	OSeq         uint32
	Seq          uint32
	OSeqHi       uint32
	SeqHi        uint32
	ReplayWindow uint32
	Bmp          []uint32
}

func (msg *XfrmReplayStateEsn) Serialize() []byte {
	// We deliberately do not pass Bmp, as it gets set by the kernel.
	return (*(*[SizeofXfrmReplayStateEsn]byte)(unsafe.Pointer(msg)))[:]
}
