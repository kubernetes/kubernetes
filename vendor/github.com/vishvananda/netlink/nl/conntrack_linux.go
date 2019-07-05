package nl

import "unsafe"

// Track the message sizes for the correct serialization/deserialization
const (
	SizeofNfgenmsg      = 4
	SizeofNfattr        = 4
	SizeofNfConntrack   = 376
	SizeofNfctTupleHead = 52
)

var L4ProtoMap = map[uint8]string{
	6:  "tcp",
	17: "udp",
}

// All the following constants are coming from:
// https://github.com/torvalds/linux/blob/master/include/uapi/linux/netfilter/nfnetlink_conntrack.h

// enum cntl_msg_types {
// 	IPCTNL_MSG_CT_NEW,
// 	IPCTNL_MSG_CT_GET,
// 	IPCTNL_MSG_CT_DELETE,
// 	IPCTNL_MSG_CT_GET_CTRZERO,
// 	IPCTNL_MSG_CT_GET_STATS_CPU,
// 	IPCTNL_MSG_CT_GET_STATS,
// 	IPCTNL_MSG_CT_GET_DYING,
// 	IPCTNL_MSG_CT_GET_UNCONFIRMED,
//
// 	IPCTNL_MSG_MAX
// };
const (
	IPCTNL_MSG_CT_GET    = 1
	IPCTNL_MSG_CT_DELETE = 2
)

// #define NFNETLINK_V0	0
const (
	NFNETLINK_V0 = 0
)

// #define NLA_F_NESTED (1 << 15)
const (
	NLA_F_NESTED = (1 << 15)
)

// enum ctattr_type {
// 	CTA_UNSPEC,
// 	CTA_TUPLE_ORIG,
// 	CTA_TUPLE_REPLY,
// 	CTA_STATUS,
// 	CTA_PROTOINFO,
// 	CTA_HELP,
// 	CTA_NAT_SRC,
// #define CTA_NAT	CTA_NAT_SRC	/* backwards compatibility */
// 	CTA_TIMEOUT,
// 	CTA_MARK,
// 	CTA_COUNTERS_ORIG,
// 	CTA_COUNTERS_REPLY,
// 	CTA_USE,
// 	CTA_ID,
// 	CTA_NAT_DST,
// 	CTA_TUPLE_MASTER,
// 	CTA_SEQ_ADJ_ORIG,
// 	CTA_NAT_SEQ_ADJ_ORIG	= CTA_SEQ_ADJ_ORIG,
// 	CTA_SEQ_ADJ_REPLY,
// 	CTA_NAT_SEQ_ADJ_REPLY	= CTA_SEQ_ADJ_REPLY,
// 	CTA_SECMARK,		/* obsolete */
// 	CTA_ZONE,
// 	CTA_SECCTX,
// 	CTA_TIMESTAMP,
// 	CTA_MARK_MASK,
// 	CTA_LABELS,
// 	CTA_LABELS_MASK,
// 	__CTA_MAX
// };
const (
	CTA_TUPLE_ORIG  = 1
	CTA_TUPLE_REPLY = 2
	CTA_STATUS      = 3
	CTA_TIMEOUT     = 7
	CTA_MARK        = 8
	CTA_PROTOINFO   = 4
)

// enum ctattr_tuple {
// 	CTA_TUPLE_UNSPEC,
// 	CTA_TUPLE_IP,
// 	CTA_TUPLE_PROTO,
// 	CTA_TUPLE_ZONE,
// 	__CTA_TUPLE_MAX
// };
// #define CTA_TUPLE_MAX (__CTA_TUPLE_MAX - 1)
const (
	CTA_TUPLE_IP    = 1
	CTA_TUPLE_PROTO = 2
)

// enum ctattr_ip {
// 	CTA_IP_UNSPEC,
// 	CTA_IP_V4_SRC,
// 	CTA_IP_V4_DST,
// 	CTA_IP_V6_SRC,
// 	CTA_IP_V6_DST,
// 	__CTA_IP_MAX
// };
// #define CTA_IP_MAX (__CTA_IP_MAX - 1)
const (
	CTA_IP_V4_SRC = 1
	CTA_IP_V4_DST = 2
	CTA_IP_V6_SRC = 3
	CTA_IP_V6_DST = 4
)

// enum ctattr_l4proto {
// 	CTA_PROTO_UNSPEC,
// 	CTA_PROTO_NUM,
// 	CTA_PROTO_SRC_PORT,
// 	CTA_PROTO_DST_PORT,
// 	CTA_PROTO_ICMP_ID,
// 	CTA_PROTO_ICMP_TYPE,
// 	CTA_PROTO_ICMP_CODE,
// 	CTA_PROTO_ICMPV6_ID,
// 	CTA_PROTO_ICMPV6_TYPE,
// 	CTA_PROTO_ICMPV6_CODE,
// 	__CTA_PROTO_MAX
// };
// #define CTA_PROTO_MAX (__CTA_PROTO_MAX - 1)
const (
	CTA_PROTO_NUM      = 1
	CTA_PROTO_SRC_PORT = 2
	CTA_PROTO_DST_PORT = 3
)

// enum ctattr_protoinfo {
// 	CTA_PROTOINFO_UNSPEC,
// 	CTA_PROTOINFO_TCP,
// 	CTA_PROTOINFO_DCCP,
// 	CTA_PROTOINFO_SCTP,
// 	__CTA_PROTOINFO_MAX
// };
// #define CTA_PROTOINFO_MAX (__CTA_PROTOINFO_MAX - 1)
const (
	CTA_PROTOINFO_TCP = 1
)

// enum ctattr_protoinfo_tcp {
// 	CTA_PROTOINFO_TCP_UNSPEC,
// 	CTA_PROTOINFO_TCP_STATE,
// 	CTA_PROTOINFO_TCP_WSCALE_ORIGINAL,
// 	CTA_PROTOINFO_TCP_WSCALE_REPLY,
// 	CTA_PROTOINFO_TCP_FLAGS_ORIGINAL,
// 	CTA_PROTOINFO_TCP_FLAGS_REPLY,
// 	__CTA_PROTOINFO_TCP_MAX
// };
// #define CTA_PROTOINFO_TCP_MAX (__CTA_PROTOINFO_TCP_MAX - 1)
const (
	CTA_PROTOINFO_TCP_STATE           = 1
	CTA_PROTOINFO_TCP_WSCALE_ORIGINAL = 2
	CTA_PROTOINFO_TCP_WSCALE_REPLY    = 3
	CTA_PROTOINFO_TCP_FLAGS_ORIGINAL  = 4
	CTA_PROTOINFO_TCP_FLAGS_REPLY     = 5
)

// /* General form of address family dependent message.
//  */
// struct nfgenmsg {
// 	__u8  nfgen_family;		/* AF_xxx */
// 	__u8  version;		/* nfnetlink version */
// 	__be16    res_id;		/* resource id */
// };
type Nfgenmsg struct {
	NfgenFamily uint8
	Version     uint8
	ResId       uint16 // big endian
}

func (msg *Nfgenmsg) Len() int {
	return SizeofNfgenmsg
}

func DeserializeNfgenmsg(b []byte) *Nfgenmsg {
	return (*Nfgenmsg)(unsafe.Pointer(&b[0:SizeofNfgenmsg][0]))
}

func (msg *Nfgenmsg) Serialize() []byte {
	return (*(*[SizeofNfgenmsg]byte)(unsafe.Pointer(msg)))[:]
}
