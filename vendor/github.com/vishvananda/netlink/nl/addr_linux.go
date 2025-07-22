package nl

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

type IfAddrmsg struct {
	unix.IfAddrmsg
}

func NewIfAddrmsg(family int) *IfAddrmsg {
	return &IfAddrmsg{
		IfAddrmsg: unix.IfAddrmsg{
			Family: uint8(family),
		},
	}
}

// struct ifaddrmsg {
//   __u8    ifa_family;
//   __u8    ifa_prefixlen;  /* The prefix length    */
//   __u8    ifa_flags;  /* Flags      */
//   __u8    ifa_scope;  /* Address scope    */
//   __u32   ifa_index;  /* Link index     */
// };

// type IfAddrmsg struct {
// 	Family    uint8
// 	Prefixlen uint8
// 	Flags     uint8
// 	Scope     uint8
// 	Index     uint32
// }
// SizeofIfAddrmsg     = 0x8

func DeserializeIfAddrmsg(b []byte) *IfAddrmsg {
	return (*IfAddrmsg)(unsafe.Pointer(&b[0:unix.SizeofIfAddrmsg][0]))
}

func (msg *IfAddrmsg) Serialize() []byte {
	return (*(*[unix.SizeofIfAddrmsg]byte)(unsafe.Pointer(msg)))[:]
}

func (msg *IfAddrmsg) Len() int {
	return unix.SizeofIfAddrmsg
}

// struct ifa_cacheinfo {
// 	__u32	ifa_prefered;
// 	__u32	ifa_valid;
// 	__u32	cstamp; /* created timestamp, hundredths of seconds */
// 	__u32	tstamp; /* updated timestamp, hundredths of seconds */
// };

type IfaCacheInfo struct {
	unix.IfaCacheinfo
}

func (msg *IfaCacheInfo) Len() int {
	return unix.SizeofIfaCacheinfo
}

func DeserializeIfaCacheInfo(b []byte) *IfaCacheInfo {
	return (*IfaCacheInfo)(unsafe.Pointer(&b[0:unix.SizeofIfaCacheinfo][0]))
}

func (msg *IfaCacheInfo) Serialize() []byte {
	return (*(*[unix.SizeofIfaCacheinfo]byte)(unsafe.Pointer(msg)))[:]
}
