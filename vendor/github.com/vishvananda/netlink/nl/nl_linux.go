// Package nl has low level primitives for making Netlink calls.
package nl

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net"
	"sync/atomic"
	"syscall"
	"unsafe"
)

const (
	// Family type definitions
	FAMILY_ALL = syscall.AF_UNSPEC
	FAMILY_V4  = syscall.AF_INET
	FAMILY_V6  = syscall.AF_INET6
)

var nextSeqNr uint32

// GetIPFamily returns the family type of a net.IP.
func GetIPFamily(ip net.IP) int {
	if len(ip) <= net.IPv4len {
		return FAMILY_V4
	}
	if ip.To4() != nil {
		return FAMILY_V4
	}
	return FAMILY_V6
}

var nativeEndian binary.ByteOrder

// Get native endianness for the system
func NativeEndian() binary.ByteOrder {
	if nativeEndian == nil {
		var x uint32 = 0x01020304
		if *(*byte)(unsafe.Pointer(&x)) == 0x01 {
			nativeEndian = binary.BigEndian
		} else {
			nativeEndian = binary.LittleEndian
		}
	}
	return nativeEndian
}

// Byte swap a 16 bit value if we aren't big endian
func Swap16(i uint16) uint16 {
	if NativeEndian() == binary.BigEndian {
		return i
	}
	return (i&0xff00)>>8 | (i&0xff)<<8
}

// Byte swap a 32 bit value if aren't big endian
func Swap32(i uint32) uint32 {
	if NativeEndian() == binary.BigEndian {
		return i
	}
	return (i&0xff000000)>>24 | (i&0xff0000)>>8 | (i&0xff00)<<8 | (i&0xff)<<24
}

type NetlinkRequestData interface {
	Len() int
	Serialize() []byte
}

// IfInfomsg is related to links, but it is used for list requests as well
type IfInfomsg struct {
	syscall.IfInfomsg
}

// Create an IfInfomsg with family specified
func NewIfInfomsg(family int) *IfInfomsg {
	return &IfInfomsg{
		IfInfomsg: syscall.IfInfomsg{
			Family: uint8(family),
		},
	}
}

func DeserializeIfInfomsg(b []byte) *IfInfomsg {
	return (*IfInfomsg)(unsafe.Pointer(&b[0:syscall.SizeofIfInfomsg][0]))
}

func (msg *IfInfomsg) Serialize() []byte {
	return (*(*[syscall.SizeofIfInfomsg]byte)(unsafe.Pointer(msg)))[:]
}

func (msg *IfInfomsg) Len() int {
	return syscall.SizeofIfInfomsg
}

func rtaAlignOf(attrlen int) int {
	return (attrlen + syscall.RTA_ALIGNTO - 1) & ^(syscall.RTA_ALIGNTO - 1)
}

func NewIfInfomsgChild(parent *RtAttr, family int) *IfInfomsg {
	msg := NewIfInfomsg(family)
	parent.children = append(parent.children, msg)
	return msg
}

// Extend RtAttr to handle data and children
type RtAttr struct {
	syscall.RtAttr
	Data     []byte
	children []NetlinkRequestData
}

// Create a new Extended RtAttr object
func NewRtAttr(attrType int, data []byte) *RtAttr {
	return &RtAttr{
		RtAttr: syscall.RtAttr{
			Type: uint16(attrType),
		},
		children: []NetlinkRequestData{},
		Data:     data,
	}
}

// Create a new RtAttr obj anc add it as a child of an existing object
func NewRtAttrChild(parent *RtAttr, attrType int, data []byte) *RtAttr {
	attr := NewRtAttr(attrType, data)
	parent.children = append(parent.children, attr)
	return attr
}

func (a *RtAttr) Len() int {
	if len(a.children) == 0 {
		return (syscall.SizeofRtAttr + len(a.Data))
	}

	l := 0
	for _, child := range a.children {
		l += rtaAlignOf(child.Len())
	}
	l += syscall.SizeofRtAttr
	return rtaAlignOf(l + len(a.Data))
}

// Serialize the RtAttr into a byte array
// This can't just unsafe.cast because it must iterate through children.
func (a *RtAttr) Serialize() []byte {
	native := NativeEndian()

	length := a.Len()
	buf := make([]byte, rtaAlignOf(length))

	if a.Data != nil {
		copy(buf[4:], a.Data)
	} else {
		next := 4
		for _, child := range a.children {
			childBuf := child.Serialize()
			copy(buf[next:], childBuf)
			next += rtaAlignOf(len(childBuf))
		}
	}

	if l := uint16(length); l != 0 {
		native.PutUint16(buf[0:2], l)
	}
	native.PutUint16(buf[2:4], a.Type)
	return buf
}

type NetlinkRequest struct {
	syscall.NlMsghdr
	Data []NetlinkRequestData
}

// Serialize the Netlink Request into a byte array
func (req *NetlinkRequest) Serialize() []byte {
	length := syscall.SizeofNlMsghdr
	dataBytes := make([][]byte, len(req.Data))
	for i, data := range req.Data {
		dataBytes[i] = data.Serialize()
		length = length + len(dataBytes[i])
	}
	req.Len = uint32(length)
	b := make([]byte, length)
	hdr := (*(*[syscall.SizeofNlMsghdr]byte)(unsafe.Pointer(req)))[:]
	next := syscall.SizeofNlMsghdr
	copy(b[0:next], hdr)
	for _, data := range dataBytes {
		for _, dataByte := range data {
			b[next] = dataByte
			next = next + 1
		}
	}
	return b
}

func (req *NetlinkRequest) AddData(data NetlinkRequestData) {
	if data != nil {
		req.Data = append(req.Data, data)
	}
}

// Execute the request against a the given sockType.
// Returns a list of netlink messages in seriaized format, optionally filtered
// by resType.
func (req *NetlinkRequest) Execute(sockType int, resType uint16) ([][]byte, error) {
	s, err := getNetlinkSocket(sockType)
	if err != nil {
		return nil, err
	}
	defer s.Close()

	if err := s.Send(req); err != nil {
		return nil, err
	}

	pid, err := s.GetPid()
	if err != nil {
		return nil, err
	}

	var res [][]byte

done:
	for {
		msgs, err := s.Receive()
		if err != nil {
			return nil, err
		}
		for _, m := range msgs {
			if m.Header.Seq != req.Seq {
				return nil, fmt.Errorf("Wrong Seq nr %d, expected 1", m.Header.Seq)
			}
			if m.Header.Pid != pid {
				return nil, fmt.Errorf("Wrong pid %d, expected %d", m.Header.Pid, pid)
			}
			if m.Header.Type == syscall.NLMSG_DONE {
				break done
			}
			if m.Header.Type == syscall.NLMSG_ERROR {
				native := NativeEndian()
				error := int32(native.Uint32(m.Data[0:4]))
				if error == 0 {
					break done
				}
				return nil, syscall.Errno(-error)
			}
			if resType != 0 && m.Header.Type != resType {
				continue
			}
			res = append(res, m.Data)
			if m.Header.Flags&syscall.NLM_F_MULTI == 0 {
				break done
			}
		}
	}
	return res, nil
}

// Create a new netlink request from proto and flags
// Note the Len value will be inaccurate once data is added until
// the message is serialized
func NewNetlinkRequest(proto, flags int) *NetlinkRequest {
	return &NetlinkRequest{
		NlMsghdr: syscall.NlMsghdr{
			Len:   uint32(syscall.SizeofNlMsghdr),
			Type:  uint16(proto),
			Flags: syscall.NLM_F_REQUEST | uint16(flags),
			Seq:   atomic.AddUint32(&nextSeqNr, 1),
		},
	}
}

type NetlinkSocket struct {
	fd  int
	lsa syscall.SockaddrNetlink
}

func getNetlinkSocket(protocol int) (*NetlinkSocket, error) {
	fd, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_RAW, protocol)
	if err != nil {
		return nil, err
	}
	s := &NetlinkSocket{
		fd: fd,
	}
	s.lsa.Family = syscall.AF_NETLINK
	if err := syscall.Bind(fd, &s.lsa); err != nil {
		syscall.Close(fd)
		return nil, err
	}

	return s, nil
}

// Create a netlink socket with a given protocol (e.g. NETLINK_ROUTE)
// and subscribe it to multicast groups passed in variable argument list.
// Returns the netlink socket on which Receive() method can be called
// to retrieve the messages from the kernel.
func Subscribe(protocol int, groups ...uint) (*NetlinkSocket, error) {
	fd, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_RAW, protocol)
	if err != nil {
		return nil, err
	}
	s := &NetlinkSocket{
		fd: fd,
	}
	s.lsa.Family = syscall.AF_NETLINK

	for _, g := range groups {
		s.lsa.Groups |= (1 << (g - 1))
	}

	if err := syscall.Bind(fd, &s.lsa); err != nil {
		syscall.Close(fd)
		return nil, err
	}

	return s, nil
}

func (s *NetlinkSocket) Close() {
	syscall.Close(s.fd)
}

func (s *NetlinkSocket) Send(request *NetlinkRequest) error {
	if err := syscall.Sendto(s.fd, request.Serialize(), 0, &s.lsa); err != nil {
		return err
	}
	return nil
}

func (s *NetlinkSocket) Receive() ([]syscall.NetlinkMessage, error) {
	rb := make([]byte, syscall.Getpagesize())
	nr, _, err := syscall.Recvfrom(s.fd, rb, 0)
	if err != nil {
		return nil, err
	}
	if nr < syscall.NLMSG_HDRLEN {
		return nil, fmt.Errorf("Got short response from netlink")
	}
	rb = rb[:nr]
	return syscall.ParseNetlinkMessage(rb)
}

func (s *NetlinkSocket) GetPid() (uint32, error) {
	lsa, err := syscall.Getsockname(s.fd)
	if err != nil {
		return 0, err
	}
	switch v := lsa.(type) {
	case *syscall.SockaddrNetlink:
		return v.Pid, nil
	}
	return 0, fmt.Errorf("Wrong socket type")
}

func ZeroTerminated(s string) []byte {
	bytes := make([]byte, len(s)+1)
	for i := 0; i < len(s); i++ {
		bytes[i] = s[i]
	}
	bytes[len(s)] = 0
	return bytes
}

func NonZeroTerminated(s string) []byte {
	bytes := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		bytes[i] = s[i]
	}
	return bytes
}

func BytesToString(b []byte) string {
	n := bytes.Index(b, []byte{0})
	return string(b[:n])
}

func Uint8Attr(v uint8) []byte {
	return []byte{byte(v)}
}

func Uint16Attr(v uint16) []byte {
	native := NativeEndian()
	bytes := make([]byte, 2)
	native.PutUint16(bytes, v)
	return bytes
}

func Uint32Attr(v uint32) []byte {
	native := NativeEndian()
	bytes := make([]byte, 4)
	native.PutUint32(bytes, v)
	return bytes
}

func ParseRouteAttr(b []byte) ([]syscall.NetlinkRouteAttr, error) {
	var attrs []syscall.NetlinkRouteAttr
	for len(b) >= syscall.SizeofRtAttr {
		a, vbuf, alen, err := netlinkRouteAttrAndValue(b)
		if err != nil {
			return nil, err
		}
		ra := syscall.NetlinkRouteAttr{Attr: *a, Value: vbuf[:int(a.Len)-syscall.SizeofRtAttr]}
		attrs = append(attrs, ra)
		b = b[alen:]
	}
	return attrs, nil
}

func netlinkRouteAttrAndValue(b []byte) (*syscall.RtAttr, []byte, int, error) {
	a := (*syscall.RtAttr)(unsafe.Pointer(&b[0]))
	if int(a.Len) < syscall.SizeofRtAttr || int(a.Len) > len(b) {
		return nil, nil, 0, syscall.EINVAL
	}
	return a, b[syscall.SizeofRtAttr:], rtaAlignOf(int(a.Len)), nil
}
