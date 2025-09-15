package nl

import (
	"bytes"
	"net"
	"unsafe"
)

// Infinity for packet and byte counts
const (
	XFRM_INF = ^uint64(0)
)

type XfrmMsgType uint8

type XfrmMsg interface {
	Type() XfrmMsgType
}

// Message Types
const (
	XFRM_MSG_BASE        XfrmMsgType = 0x10
	XFRM_MSG_NEWSA                   = 0x10
	XFRM_MSG_DELSA                   = 0x11
	XFRM_MSG_GETSA                   = 0x12
	XFRM_MSG_NEWPOLICY               = 0x13
	XFRM_MSG_DELPOLICY               = 0x14
	XFRM_MSG_GETPOLICY               = 0x15
	XFRM_MSG_ALLOCSPI                = 0x16
	XFRM_MSG_ACQUIRE                 = 0x17
	XFRM_MSG_EXPIRE                  = 0x18
	XFRM_MSG_UPDPOLICY               = 0x19
	XFRM_MSG_UPDSA                   = 0x1a
	XFRM_MSG_POLEXPIRE               = 0x1b
	XFRM_MSG_FLUSHSA                 = 0x1c
	XFRM_MSG_FLUSHPOLICY             = 0x1d
	XFRM_MSG_NEWAE                   = 0x1e
	XFRM_MSG_GETAE                   = 0x1f
	XFRM_MSG_REPORT                  = 0x20
	XFRM_MSG_MIGRATE                 = 0x21
	XFRM_MSG_NEWSADINFO              = 0x22
	XFRM_MSG_GETSADINFO              = 0x23
	XFRM_MSG_NEWSPDINFO              = 0x24
	XFRM_MSG_GETSPDINFO              = 0x25
	XFRM_MSG_MAPPING                 = 0x26
	XFRM_MSG_MAX                     = 0x26
	XFRM_NR_MSGTYPES                 = 0x17
)

// Attribute types
const (
	/* Netlink message attributes.  */
	XFRMA_UNSPEC    = iota
	XFRMA_ALG_AUTH  /* struct xfrm_algo */
	XFRMA_ALG_CRYPT /* struct xfrm_algo */
	XFRMA_ALG_COMP  /* struct xfrm_algo */
	XFRMA_ENCAP     /* struct xfrm_algo + struct xfrm_encap_tmpl */
	XFRMA_TMPL      /* 1 or more struct xfrm_user_tmpl */
	XFRMA_SA        /* struct xfrm_usersa_info  */
	XFRMA_POLICY    /* struct xfrm_userpolicy_info */
	XFRMA_SEC_CTX   /* struct xfrm_sec_ctx */
	XFRMA_LTIME_VAL
	XFRMA_REPLAY_VAL
	XFRMA_REPLAY_THRESH
	XFRMA_ETIMER_THRESH
	XFRMA_SRCADDR     /* xfrm_address_t */
	XFRMA_COADDR      /* xfrm_address_t */
	XFRMA_LASTUSED    /* unsigned long  */
	XFRMA_POLICY_TYPE /* struct xfrm_userpolicy_type */
	XFRMA_MIGRATE
	XFRMA_ALG_AEAD       /* struct xfrm_algo_aead */
	XFRMA_KMADDRESS      /* struct xfrm_user_kmaddress */
	XFRMA_ALG_AUTH_TRUNC /* struct xfrm_algo_auth */
	XFRMA_MARK           /* struct xfrm_mark */
	XFRMA_TFCPAD         /* __u32 */
	XFRMA_REPLAY_ESN_VAL /* struct xfrm_replay_esn */
	XFRMA_SA_EXTRA_FLAGS /* __u32 */
	XFRMA_PROTO          /* __u8 */
	XFRMA_ADDRESS_FILTER /* struct xfrm_address_filter */
	XFRMA_PAD
	XFRMA_OFFLOAD_DEV            /* struct xfrm_state_offload */
	XFRMA_SET_MARK               /* __u32 */
	XFRMA_SET_MARK_MASK          /* __u32 */
	XFRMA_IF_ID                  /* __u32 */
	XFRMA_MTIMER_THRESH          /* __u32 in seconds for input SA */
	XFRMA_SA_DIR                 /* __u8 */
	XFRMA_NAT_KEEPALIVE_INTERVAL /* __u32 in seconds for NAT keepalive */
	XFRMA_SA_PCPU                /* __u32 */

	XFRMA_MAX = iota - 1
)

const XFRMA_OUTPUT_MARK = XFRMA_SET_MARK

const (
	SizeofXfrmAddress     = 0x10
	SizeofXfrmSelector    = 0x38
	SizeofXfrmLifetimeCfg = 0x40
	SizeofXfrmLifetimeCur = 0x20
	SizeofXfrmId          = 0x18
	SizeofXfrmMark        = 0x08
)

// Netlink groups
const (
	XFRMNLGRP_NONE    = 0x0
	XFRMNLGRP_ACQUIRE = 0x1
	XFRMNLGRP_EXPIRE  = 0x2
	XFRMNLGRP_SA      = 0x3
	XFRMNLGRP_POLICY  = 0x4
	XFRMNLGRP_AEVENTS = 0x5
	XFRMNLGRP_REPORT  = 0x6
	XFRMNLGRP_MIGRATE = 0x7
	XFRMNLGRP_MAPPING = 0x8
	__XFRMNLGRP_MAX   = 0x9
)

// typedef union {
//   __be32    a4;
//   __be32    a6[4];
// } xfrm_address_t;

type XfrmAddress [SizeofXfrmAddress]byte

func (x *XfrmAddress) ToIP() net.IP {
	var empty = [12]byte{}
	ip := make(net.IP, net.IPv6len)
	if bytes.Equal(x[4:16], empty[:]) {
		ip[10] = 0xff
		ip[11] = 0xff
		copy(ip[12:16], x[0:4])
	} else {
		copy(ip[:], x[:])
	}
	return ip
}

// family is only used when x and prefixlen are both 0
func (x *XfrmAddress) ToIPNet(prefixlen uint8, family uint16) *net.IPNet {
	empty := [SizeofXfrmAddress]byte{}
	if bytes.Equal(x[:], empty[:]) && prefixlen == 0 {
		if family == FAMILY_V6 {
			return &net.IPNet{IP: net.ParseIP("::"), Mask: net.CIDRMask(int(prefixlen), 128)}
		}
		return &net.IPNet{IP: net.ParseIP("0.0.0.0"), Mask: net.CIDRMask(int(prefixlen), 32)}
	}
	ip := x.ToIP()
	if GetIPFamily(ip) == FAMILY_V4 {
		return &net.IPNet{IP: ip, Mask: net.CIDRMask(int(prefixlen), 32)}
	}
	return &net.IPNet{IP: ip, Mask: net.CIDRMask(int(prefixlen), 128)}
}

func (x *XfrmAddress) FromIP(ip net.IP) {
	var empty = [16]byte{}
	if len(ip) < net.IPv4len {
		copy(x[4:16], empty[:])
	} else if GetIPFamily(ip) == FAMILY_V4 {
		copy(x[0:4], ip.To4()[0:4])
		copy(x[4:16], empty[:12])
	} else {
		copy(x[0:16], ip.To16()[0:16])
	}
}

func DeserializeXfrmAddress(b []byte) *XfrmAddress {
	return (*XfrmAddress)(unsafe.Pointer(&b[0:SizeofXfrmAddress][0]))
}

func (x *XfrmAddress) Serialize() []byte {
	return (*(*[SizeofXfrmAddress]byte)(unsafe.Pointer(x)))[:]
}

// struct xfrm_selector {
//   xfrm_address_t  daddr;
//   xfrm_address_t  saddr;
//   __be16  dport;
//   __be16  dport_mask;
//   __be16  sport;
//   __be16  sport_mask;
//   __u16 family;
//   __u8  prefixlen_d;
//   __u8  prefixlen_s;
//   __u8  proto;
//   int ifindex;
//   __kernel_uid32_t  user;
// };

type XfrmSelector struct {
	Daddr      XfrmAddress
	Saddr      XfrmAddress
	Dport      uint16 // big endian
	DportMask  uint16 // big endian
	Sport      uint16 // big endian
	SportMask  uint16 // big endian
	Family     uint16
	PrefixlenD uint8
	PrefixlenS uint8
	Proto      uint8
	Pad        [3]byte
	Ifindex    int32
	User       uint32
}

func (msg *XfrmSelector) Len() int {
	return SizeofXfrmSelector
}

func DeserializeXfrmSelector(b []byte) *XfrmSelector {
	return (*XfrmSelector)(unsafe.Pointer(&b[0:SizeofXfrmSelector][0]))
}

func (msg *XfrmSelector) Serialize() []byte {
	return (*(*[SizeofXfrmSelector]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_lifetime_cfg {
//   __u64 soft_byte_limit;
//   __u64 hard_byte_limit;
//   __u64 soft_packet_limit;
//   __u64 hard_packet_limit;
//   __u64 soft_add_expires_seconds;
//   __u64 hard_add_expires_seconds;
//   __u64 soft_use_expires_seconds;
//   __u64 hard_use_expires_seconds;
// };
//

type XfrmLifetimeCfg struct {
	SoftByteLimit         uint64
	HardByteLimit         uint64
	SoftPacketLimit       uint64
	HardPacketLimit       uint64
	SoftAddExpiresSeconds uint64
	HardAddExpiresSeconds uint64
	SoftUseExpiresSeconds uint64
	HardUseExpiresSeconds uint64
}

func (msg *XfrmLifetimeCfg) Len() int {
	return SizeofXfrmLifetimeCfg
}

func DeserializeXfrmLifetimeCfg(b []byte) *XfrmLifetimeCfg {
	return (*XfrmLifetimeCfg)(unsafe.Pointer(&b[0:SizeofXfrmLifetimeCfg][0]))
}

func (msg *XfrmLifetimeCfg) Serialize() []byte {
	return (*(*[SizeofXfrmLifetimeCfg]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_lifetime_cur {
//   __u64 bytes;
//   __u64 packets;
//   __u64 add_time;
//   __u64 use_time;
// };

type XfrmLifetimeCur struct {
	Bytes   uint64
	Packets uint64
	AddTime uint64
	UseTime uint64
}

func (msg *XfrmLifetimeCur) Len() int {
	return SizeofXfrmLifetimeCur
}

func DeserializeXfrmLifetimeCur(b []byte) *XfrmLifetimeCur {
	return (*XfrmLifetimeCur)(unsafe.Pointer(&b[0:SizeofXfrmLifetimeCur][0]))
}

func (msg *XfrmLifetimeCur) Serialize() []byte {
	return (*(*[SizeofXfrmLifetimeCur]byte)(unsafe.Pointer(msg)))[:]
}

// struct xfrm_id {
//   xfrm_address_t  daddr;
//   __be32    spi;
//   __u8    proto;
// };

type XfrmId struct {
	Daddr XfrmAddress
	Spi   uint32 // big endian
	Proto uint8
	Pad   [3]byte
}

func (msg *XfrmId) Len() int {
	return SizeofXfrmId
}

func DeserializeXfrmId(b []byte) *XfrmId {
	return (*XfrmId)(unsafe.Pointer(&b[0:SizeofXfrmId][0]))
}

func (msg *XfrmId) Serialize() []byte {
	return (*(*[SizeofXfrmId]byte)(unsafe.Pointer(msg)))[:]
}

type XfrmMark struct {
	Value uint32
	Mask  uint32
}

func (msg *XfrmMark) Len() int {
	return SizeofXfrmMark
}

func DeserializeXfrmMark(b []byte) *XfrmMark {
	return (*XfrmMark)(unsafe.Pointer(&b[0:SizeofXfrmMark][0]))
}

func (msg *XfrmMark) Serialize() []byte {
	return (*(*[SizeofXfrmMark]byte)(unsafe.Pointer(msg)))[:]
}
