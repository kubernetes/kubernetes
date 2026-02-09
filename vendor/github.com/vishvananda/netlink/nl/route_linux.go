package nl

import (
	"unsafe"

	"golang.org/x/sys/unix"
)

type RtMsg struct {
	unix.RtMsg
}

func NewRtMsg() *RtMsg {
	return &RtMsg{
		RtMsg: unix.RtMsg{
			Table:    unix.RT_TABLE_MAIN,
			Scope:    unix.RT_SCOPE_UNIVERSE,
			Protocol: unix.RTPROT_BOOT,
			Type:     unix.RTN_UNICAST,
		},
	}
}

func NewRtDelMsg() *RtMsg {
	return &RtMsg{
		RtMsg: unix.RtMsg{
			Table: unix.RT_TABLE_MAIN,
			Scope: unix.RT_SCOPE_NOWHERE,
		},
	}
}

func (msg *RtMsg) Len() int {
	return unix.SizeofRtMsg
}

func DeserializeRtMsg(b []byte) *RtMsg {
	return (*RtMsg)(unsafe.Pointer(&b[0:unix.SizeofRtMsg][0]))
}

func (msg *RtMsg) Serialize() []byte {
	return (*(*[unix.SizeofRtMsg]byte)(unsafe.Pointer(msg)))[:]
}

type RtNexthop struct {
	unix.RtNexthop
	Children []NetlinkRequestData
}

func DeserializeRtNexthop(b []byte) *RtNexthop {
	return &RtNexthop{
		RtNexthop: *((*unix.RtNexthop)(unsafe.Pointer(&b[0:unix.SizeofRtNexthop][0]))),
	}
}

func (msg *RtNexthop) Len() int {
	if len(msg.Children) == 0 {
		return unix.SizeofRtNexthop
	}

	l := 0
	for _, child := range msg.Children {
		l += rtaAlignOf(child.Len())
	}
	l += unix.SizeofRtNexthop
	return rtaAlignOf(l)
}

func (msg *RtNexthop) Serialize() []byte {
	length := msg.Len()
	msg.RtNexthop.Len = uint16(length)
	buf := make([]byte, length)
	copy(buf, (*(*[unix.SizeofRtNexthop]byte)(unsafe.Pointer(msg)))[:])
	next := rtaAlignOf(unix.SizeofRtNexthop)
	if len(msg.Children) > 0 {
		for _, child := range msg.Children {
			childBuf := child.Serialize()
			copy(buf[next:], childBuf)
			next += rtaAlignOf(len(childBuf))
		}
	}
	return buf
}

type RtGenMsg struct {
	unix.RtGenmsg
}

func NewRtGenMsg() *RtGenMsg {
	return &RtGenMsg{
		RtGenmsg: unix.RtGenmsg{
			Family: unix.AF_UNSPEC,
		},
	}
}

func (msg *RtGenMsg) Len() int {
	return rtaAlignOf(unix.SizeofRtGenmsg)
}

func DeserializeRtGenMsg(b []byte) *RtGenMsg {
	return &RtGenMsg{RtGenmsg: unix.RtGenmsg{Family: b[0]}}
}

func (msg *RtGenMsg) Serialize() []byte {
	out := make([]byte, msg.Len())
	out[0] = msg.Family
	return out
}
