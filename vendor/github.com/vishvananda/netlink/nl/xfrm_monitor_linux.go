package nl

import (
	"unsafe"
)

const (
	SizeofXfrmUserExpire = 0xe8
)

// struct xfrm_user_expire {
// 	struct xfrm_usersa_info		state;
// 	__u8				hard;
// };

type XfrmUserExpire struct {
	XfrmUsersaInfo XfrmUsersaInfo
	Hard           uint8
	Pad            [7]byte
}

func (msg *XfrmUserExpire) Len() int {
	return SizeofXfrmUserExpire
}

func DeserializeXfrmUserExpire(b []byte) *XfrmUserExpire {
	return (*XfrmUserExpire)(unsafe.Pointer(&b[0:SizeofXfrmUserExpire][0]))
}

func (msg *XfrmUserExpire) Serialize() []byte {
	return (*(*[SizeofXfrmUserExpire]byte)(unsafe.Pointer(msg)))[:]
}
