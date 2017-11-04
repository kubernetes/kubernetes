package nl

import (
	"syscall"
	"unsafe"
)

type RtMsg struct {
	syscall.RtMsg
}

func NewRtMsg() *RtMsg {
	return &RtMsg{
		RtMsg: syscall.RtMsg{
			Table:    syscall.RT_TABLE_MAIN,
			Scope:    syscall.RT_SCOPE_UNIVERSE,
			Protocol: syscall.RTPROT_BOOT,
			Type:     syscall.RTN_UNICAST,
		},
	}
}

func NewRtDelMsg() *RtMsg {
	return &RtMsg{
		RtMsg: syscall.RtMsg{
			Table: syscall.RT_TABLE_MAIN,
			Scope: syscall.RT_SCOPE_NOWHERE,
		},
	}
}

func (msg *RtMsg) Len() int {
	return syscall.SizeofRtMsg
}

func DeserializeRtMsg(b []byte) *RtMsg {
	return (*RtMsg)(unsafe.Pointer(&b[0:syscall.SizeofRtMsg][0]))
}

func (msg *RtMsg) Serialize() []byte {
	return (*(*[syscall.SizeofRtMsg]byte)(unsafe.Pointer(msg)))[:]
}

type RtNexthop struct {
	syscall.RtNexthop
	Children []NetlinkRequestData
}

func DeserializeRtNexthop(b []byte) *RtNexthop {
	return (*RtNexthop)(unsafe.Pointer(&b[0:syscall.SizeofRtNexthop][0]))
}

func (msg *RtNexthop) Len() int {
	if len(msg.Children) == 0 {
		return syscall.SizeofRtNexthop
	}

	l := 0
	for _, child := range msg.Children {
		l += rtaAlignOf(child.Len())
	}
	l += syscall.SizeofRtNexthop
	return rtaAlignOf(l)
}

func (msg *RtNexthop) Serialize() []byte {
	length := msg.Len()
	msg.RtNexthop.Len = uint16(length)
	buf := make([]byte, length)
	copy(buf, (*(*[syscall.SizeofRtNexthop]byte)(unsafe.Pointer(msg)))[:])
	next := rtaAlignOf(syscall.SizeofRtNexthop)
	if len(msg.Children) > 0 {
		for _, child := range msg.Children {
			childBuf := child.Serialize()
			copy(buf[next:], childBuf)
			next += rtaAlignOf(len(childBuf))
		}
	}
	return buf
}
