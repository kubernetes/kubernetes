// +build linux

package libcontainer

import (
	"syscall"

	"github.com/vishvananda/netlink/nl"
)

// list of known message types we want to send to bootstrap program
// The number is randomly chosen to not conflict with known netlink types
const (
	InitMsg         uint16 = 62000
	CloneFlagsAttr  uint16 = 27281
	ConsolePathAttr uint16 = 27282
	NsPathsAttr     uint16 = 27283
	UidmapAttr      uint16 = 27284
	GidmapAttr      uint16 = 27285
	SetgroupAttr    uint16 = 27286
	// When syscall.NLA_HDRLEN is in gccgo, take this out.
	syscall_NLA_HDRLEN = (syscall.SizeofNlAttr + syscall.NLA_ALIGNTO - 1) & ^(syscall.NLA_ALIGNTO - 1)
)

type Int32msg struct {
	Type  uint16
	Value uint32
}

// Serialize serializes the message.
// Int32msg has the following representation
// | nlattr len | nlattr type |
// | uint32 value             |
func (msg *Int32msg) Serialize() []byte {
	buf := make([]byte, msg.Len())
	native := nl.NativeEndian()
	native.PutUint16(buf[0:2], uint16(msg.Len()))
	native.PutUint16(buf[2:4], msg.Type)
	native.PutUint32(buf[4:8], msg.Value)
	return buf
}

func (msg *Int32msg) Len() int {
	return syscall_NLA_HDRLEN + 4
}

// Bytemsg has the following representation
// | nlattr len | nlattr type |
// | value              | pad |
type Bytemsg struct {
	Type  uint16
	Value []byte
}

func (msg *Bytemsg) Serialize() []byte {
	l := msg.Len()
	buf := make([]byte, (l+syscall.NLA_ALIGNTO-1) & ^(syscall.NLA_ALIGNTO-1))
	native := nl.NativeEndian()
	native.PutUint16(buf[0:2], uint16(l))
	native.PutUint16(buf[2:4], msg.Type)
	copy(buf[4:], msg.Value)
	return buf
}

func (msg *Bytemsg) Len() int {
	return syscall_NLA_HDRLEN + len(msg.Value) + 1 // null-terminated
}

type Boolmsg struct {
	Type  uint16
	Value bool
}

func (msg *Boolmsg) Serialize() []byte {
	buf := make([]byte, msg.Len())
	native := nl.NativeEndian()
	native.PutUint16(buf[0:2], uint16(msg.Len()))
	native.PutUint16(buf[2:4], msg.Type)
	if msg.Value {
		buf[4] = 1
	} else {
		buf[4] = 0
	}
	return buf
}

func (msg *Boolmsg) Len() int {
	return syscall_NLA_HDRLEN + 1
}
