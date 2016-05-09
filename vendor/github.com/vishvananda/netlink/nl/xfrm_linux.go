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

// Message Types
const (
	XFRM_MSG_BASE        = 0x10
	XFRM_MSG_NEWSA       = 0x10
	XFRM_MSG_DELSA       = 0x11
	XFRM_MSG_GETSA       = 0x12
	XFRM_MSG_NEWPOLICY   = 0x13
	XFRM_MSG_DELPOLICY   = 0x14
	XFRM_MSG_GETPOLICY   = 0x15
	XFRM_MSG_ALLOCSPI    = 0x16
	XFRM_MSG_ACQUIRE     = 0x17
	XFRM_MSG_EXPIRE      = 0x18
	XFRM_MSG_UPDPOLICY   = 0x19
	XFRM_MSG_UPDSA       = 0x1a
	XFRM_MSG_POLEXPIRE   = 0x1b
	XFRM_MSG_FLUSHSA     = 0x1c
	XFRM_MSG_FLUSHPOLICY = 0x1d
	XFRM_MSG_NEWAE       = 0x1e
	XFRM_MSG_GETAE       = 0x1f
	XFRM_MSG_REPORT      = 0x20
	XFRM_MSG_MIGRATE     = 0x21
	XFRM_MSG_NEWSADINFO  = 0x22
	XFRM_MSG_GETSADINFO  = 0x23
	XFRM_MSG_NEWSPDINFO  = 0x24
	XFRM_MSG_GETSPDINFO  = 0x25
	XFRM_MSG_MAPPING     = 0x26
	XFRM_MSG_MAX         = 0x26
	XFRM_NR_MSGTYPES     = 0x17
)

// Attribute types
const (
	/* Netlink message attributes.  */
	XFRMA_UNSPEC         = 0x00
	XFRMA_ALG_AUTH       = 0x01 /* struct xfrm_algo */
	XFRMA_ALG_CRYPT      = 0x02 /* struct xfrm_algo */
	XFRMA_ALG_COMP       = 0x03 /* struct xfrm_algo */
	XFRMA_ENCAP          = 0x04 /* struct xfrm_algo + struct xfrm_encap_tmpl */
	XFRMA_TMPL           = 0x05 /* 1 or more struct xfrm_user_tmpl */
	XFRMA_SA             = 0x06 /* struct xfrm_usersa_info  */
	XFRMA_POLICY         = 0x07 /* struct xfrm_userpolicy_info */
	XFRMA_SEC_CTX        = 0x08 /* struct xfrm_sec_ctx */
	XFRMA_LTIME_VAL      = 0x09
	XFRMA_REPLAY_VAL     = 0x0a
	XFRMA_REPLAY_THRESH  = 0x0b
	XFRMA_ETIMER_THRESH  = 0x0c
	XFRMA_SRCADDR        = 0x0d /* xfrm_address_t */
	XFRMA_COADDR         = 0x0e /* xfrm_address_t */
	XFRMA_LASTUSED       = 0x0f /* unsigned long  */
	XFRMA_POLICY_TYPE    = 0x10 /* struct xfrm_userpolicy_type */
	XFRMA_MIGRATE        = 0x11
	XFRMA_ALG_AEAD       = 0x12 /* struct xfrm_algo_aead */
	XFRMA_KMADDRESS      = 0x13 /* struct xfrm_user_kmaddress */
	XFRMA_ALG_AUTH_TRUNC = 0x14 /* struct xfrm_algo_auth */
	XFRMA_MARK           = 0x15 /* struct xfrm_mark */
	XFRMA_TFCPAD         = 0x16 /* __u32 */
	XFRMA_REPLAY_ESN_VAL = 0x17 /* struct xfrm_replay_esn */
	XFRMA_SA_EXTRA_FLAGS = 0x18 /* __u32 */
	XFRMA_MAX            = 0x18
)

const (
	SizeofXfrmAddress     = 0x10
	SizeofXfrmSelector    = 0x38
	SizeofXfrmLifetimeCfg = 0x40
	SizeofXfrmLifetimeCur = 0x20
	SizeofXfrmId          = 0x18
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

func (x *XfrmAddress) ToIPNet(prefixlen uint8) *net.IPNet {
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
