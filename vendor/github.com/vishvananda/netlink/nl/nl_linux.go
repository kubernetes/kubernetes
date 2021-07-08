// Package nl has low level primitives for making Netlink calls.
package nl

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
	"unsafe"

	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

const (
	// Family type definitions
	FAMILY_ALL  = unix.AF_UNSPEC
	FAMILY_V4   = unix.AF_INET
	FAMILY_V6   = unix.AF_INET6
	FAMILY_MPLS = unix.AF_MPLS
	// Arbitrary set value (greater than default 4k) to allow receiving
	// from kernel more verbose messages e.g. for statistics,
	// tc rules or filters, or other more memory requiring data.
	RECEIVE_BUFFER_SIZE = 65536
	// Kernel netlink pid
	PidKernel uint32 = 0
)

// SupportedNlFamilies contains the list of netlink families this netlink package supports
var SupportedNlFamilies = []int{unix.NETLINK_ROUTE, unix.NETLINK_XFRM, unix.NETLINK_NETFILTER}

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

// NativeEndian gets native endianness for the system
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
	unix.IfInfomsg
}

// Create an IfInfomsg with family specified
func NewIfInfomsg(family int) *IfInfomsg {
	return &IfInfomsg{
		IfInfomsg: unix.IfInfomsg{
			Family: uint8(family),
		},
	}
}

func DeserializeIfInfomsg(b []byte) *IfInfomsg {
	return (*IfInfomsg)(unsafe.Pointer(&b[0:unix.SizeofIfInfomsg][0]))
}

func (msg *IfInfomsg) Serialize() []byte {
	return (*(*[unix.SizeofIfInfomsg]byte)(unsafe.Pointer(msg)))[:]
}

func (msg *IfInfomsg) Len() int {
	return unix.SizeofIfInfomsg
}

func (msg *IfInfomsg) EncapType() string {
	switch msg.Type {
	case 0:
		return "generic"
	case unix.ARPHRD_ETHER:
		return "ether"
	case unix.ARPHRD_EETHER:
		return "eether"
	case unix.ARPHRD_AX25:
		return "ax25"
	case unix.ARPHRD_PRONET:
		return "pronet"
	case unix.ARPHRD_CHAOS:
		return "chaos"
	case unix.ARPHRD_IEEE802:
		return "ieee802"
	case unix.ARPHRD_ARCNET:
		return "arcnet"
	case unix.ARPHRD_APPLETLK:
		return "atalk"
	case unix.ARPHRD_DLCI:
		return "dlci"
	case unix.ARPHRD_ATM:
		return "atm"
	case unix.ARPHRD_METRICOM:
		return "metricom"
	case unix.ARPHRD_IEEE1394:
		return "ieee1394"
	case unix.ARPHRD_INFINIBAND:
		return "infiniband"
	case unix.ARPHRD_SLIP:
		return "slip"
	case unix.ARPHRD_CSLIP:
		return "cslip"
	case unix.ARPHRD_SLIP6:
		return "slip6"
	case unix.ARPHRD_CSLIP6:
		return "cslip6"
	case unix.ARPHRD_RSRVD:
		return "rsrvd"
	case unix.ARPHRD_ADAPT:
		return "adapt"
	case unix.ARPHRD_ROSE:
		return "rose"
	case unix.ARPHRD_X25:
		return "x25"
	case unix.ARPHRD_HWX25:
		return "hwx25"
	case unix.ARPHRD_PPP:
		return "ppp"
	case unix.ARPHRD_HDLC:
		return "hdlc"
	case unix.ARPHRD_LAPB:
		return "lapb"
	case unix.ARPHRD_DDCMP:
		return "ddcmp"
	case unix.ARPHRD_RAWHDLC:
		return "rawhdlc"
	case unix.ARPHRD_TUNNEL:
		return "ipip"
	case unix.ARPHRD_TUNNEL6:
		return "tunnel6"
	case unix.ARPHRD_FRAD:
		return "frad"
	case unix.ARPHRD_SKIP:
		return "skip"
	case unix.ARPHRD_LOOPBACK:
		return "loopback"
	case unix.ARPHRD_LOCALTLK:
		return "ltalk"
	case unix.ARPHRD_FDDI:
		return "fddi"
	case unix.ARPHRD_BIF:
		return "bif"
	case unix.ARPHRD_SIT:
		return "sit"
	case unix.ARPHRD_IPDDP:
		return "ip/ddp"
	case unix.ARPHRD_IPGRE:
		return "gre"
	case unix.ARPHRD_PIMREG:
		return "pimreg"
	case unix.ARPHRD_HIPPI:
		return "hippi"
	case unix.ARPHRD_ASH:
		return "ash"
	case unix.ARPHRD_ECONET:
		return "econet"
	case unix.ARPHRD_IRDA:
		return "irda"
	case unix.ARPHRD_FCPP:
		return "fcpp"
	case unix.ARPHRD_FCAL:
		return "fcal"
	case unix.ARPHRD_FCPL:
		return "fcpl"
	case unix.ARPHRD_FCFABRIC:
		return "fcfb0"
	case unix.ARPHRD_FCFABRIC + 1:
		return "fcfb1"
	case unix.ARPHRD_FCFABRIC + 2:
		return "fcfb2"
	case unix.ARPHRD_FCFABRIC + 3:
		return "fcfb3"
	case unix.ARPHRD_FCFABRIC + 4:
		return "fcfb4"
	case unix.ARPHRD_FCFABRIC + 5:
		return "fcfb5"
	case unix.ARPHRD_FCFABRIC + 6:
		return "fcfb6"
	case unix.ARPHRD_FCFABRIC + 7:
		return "fcfb7"
	case unix.ARPHRD_FCFABRIC + 8:
		return "fcfb8"
	case unix.ARPHRD_FCFABRIC + 9:
		return "fcfb9"
	case unix.ARPHRD_FCFABRIC + 10:
		return "fcfb10"
	case unix.ARPHRD_FCFABRIC + 11:
		return "fcfb11"
	case unix.ARPHRD_FCFABRIC + 12:
		return "fcfb12"
	case unix.ARPHRD_IEEE802_TR:
		return "tr"
	case unix.ARPHRD_IEEE80211:
		return "ieee802.11"
	case unix.ARPHRD_IEEE80211_PRISM:
		return "ieee802.11/prism"
	case unix.ARPHRD_IEEE80211_RADIOTAP:
		return "ieee802.11/radiotap"
	case unix.ARPHRD_IEEE802154:
		return "ieee802.15.4"

	case 65534:
		return "none"
	case 65535:
		return "void"
	}
	return fmt.Sprintf("unknown%d", msg.Type)
}

func rtaAlignOf(attrlen int) int {
	return (attrlen + unix.RTA_ALIGNTO - 1) & ^(unix.RTA_ALIGNTO - 1)
}

func NewIfInfomsgChild(parent *RtAttr, family int) *IfInfomsg {
	msg := NewIfInfomsg(family)
	parent.children = append(parent.children, msg)
	return msg
}

type Uint32Attribute struct {
	Type  uint16
	Value uint32
}

func (a *Uint32Attribute) Serialize() []byte {
	native := NativeEndian()
	buf := make([]byte, rtaAlignOf(8))
	native.PutUint16(buf[0:2], 8)
	native.PutUint16(buf[2:4], a.Type)

	if a.Type&NLA_F_NET_BYTEORDER != 0 {
		binary.BigEndian.PutUint32(buf[4:], a.Value)
	} else {
		native.PutUint32(buf[4:], a.Value)
	}
	return buf
}

func (a *Uint32Attribute) Len() int {
	return 8
}

// Extend RtAttr to handle data and children
type RtAttr struct {
	unix.RtAttr
	Data     []byte
	children []NetlinkRequestData
}

// Create a new Extended RtAttr object
func NewRtAttr(attrType int, data []byte) *RtAttr {
	return &RtAttr{
		RtAttr: unix.RtAttr{
			Type: uint16(attrType),
		},
		children: []NetlinkRequestData{},
		Data:     data,
	}
}

// NewRtAttrChild adds an RtAttr as a child to the parent and returns the new attribute
//
// Deprecated: Use AddRtAttr() on the parent object
func NewRtAttrChild(parent *RtAttr, attrType int, data []byte) *RtAttr {
	return parent.AddRtAttr(attrType, data)
}

// AddRtAttr adds an RtAttr as a child and returns the new attribute
func (a *RtAttr) AddRtAttr(attrType int, data []byte) *RtAttr {
	attr := NewRtAttr(attrType, data)
	a.children = append(a.children, attr)
	return attr
}

// AddChild adds an existing NetlinkRequestData as a child.
func (a *RtAttr) AddChild(attr NetlinkRequestData) {
	a.children = append(a.children, attr)
}

func (a *RtAttr) Len() int {
	if len(a.children) == 0 {
		return (unix.SizeofRtAttr + len(a.Data))
	}

	l := 0
	for _, child := range a.children {
		l += rtaAlignOf(child.Len())
	}
	l += unix.SizeofRtAttr
	return rtaAlignOf(l + len(a.Data))
}

// Serialize the RtAttr into a byte array
// This can't just unsafe.cast because it must iterate through children.
func (a *RtAttr) Serialize() []byte {
	native := NativeEndian()

	length := a.Len()
	buf := make([]byte, rtaAlignOf(length))

	next := 4
	if a.Data != nil {
		copy(buf[next:], a.Data)
		next += rtaAlignOf(len(a.Data))
	}
	if len(a.children) > 0 {
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
	unix.NlMsghdr
	Data    []NetlinkRequestData
	RawData []byte
	Sockets map[int]*SocketHandle
}

// Serialize the Netlink Request into a byte array
func (req *NetlinkRequest) Serialize() []byte {
	length := unix.SizeofNlMsghdr
	dataBytes := make([][]byte, len(req.Data))
	for i, data := range req.Data {
		dataBytes[i] = data.Serialize()
		length = length + len(dataBytes[i])
	}
	length += len(req.RawData)

	req.Len = uint32(length)
	b := make([]byte, length)
	hdr := (*(*[unix.SizeofNlMsghdr]byte)(unsafe.Pointer(req)))[:]
	next := unix.SizeofNlMsghdr
	copy(b[0:next], hdr)
	for _, data := range dataBytes {
		for _, dataByte := range data {
			b[next] = dataByte
			next = next + 1
		}
	}
	// Add the raw data if any
	if len(req.RawData) > 0 {
		copy(b[next:length], req.RawData)
	}
	return b
}

func (req *NetlinkRequest) AddData(data NetlinkRequestData) {
	req.Data = append(req.Data, data)
}

// AddRawData adds raw bytes to the end of the NetlinkRequest object during serialization
func (req *NetlinkRequest) AddRawData(data []byte) {
	req.RawData = append(req.RawData, data...)
}

// Execute the request against a the given sockType.
// Returns a list of netlink messages in serialized format, optionally filtered
// by resType.
func (req *NetlinkRequest) Execute(sockType int, resType uint16) ([][]byte, error) {
	var (
		s   *NetlinkSocket
		err error
	)

	if req.Sockets != nil {
		if sh, ok := req.Sockets[sockType]; ok {
			s = sh.Socket
			req.Seq = atomic.AddUint32(&sh.Seq, 1)
		}
	}
	sharedSocket := s != nil

	if s == nil {
		s, err = getNetlinkSocket(sockType)
		if err != nil {
			return nil, err
		}
		defer s.Close()
	} else {
		s.Lock()
		defer s.Unlock()
	}

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
		msgs, from, err := s.Receive()
		if err != nil {
			return nil, err
		}
		if from.Pid != PidKernel {
			return nil, fmt.Errorf("Wrong sender portid %d, expected %d", from.Pid, PidKernel)
		}
		for _, m := range msgs {
			if m.Header.Seq != req.Seq {
				if sharedSocket {
					continue
				}
				return nil, fmt.Errorf("Wrong Seq nr %d, expected %d", m.Header.Seq, req.Seq)
			}
			if m.Header.Pid != pid {
				continue
			}
			if m.Header.Type == unix.NLMSG_DONE || m.Header.Type == unix.NLMSG_ERROR {
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
			if m.Header.Flags&unix.NLM_F_MULTI == 0 {
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
		NlMsghdr: unix.NlMsghdr{
			Len:   uint32(unix.SizeofNlMsghdr),
			Type:  uint16(proto),
			Flags: unix.NLM_F_REQUEST | uint16(flags),
			Seq:   atomic.AddUint32(&nextSeqNr, 1),
		},
	}
}

type NetlinkSocket struct {
	fd  int32
	lsa unix.SockaddrNetlink
	sync.Mutex
}

func getNetlinkSocket(protocol int) (*NetlinkSocket, error) {
	fd, err := unix.Socket(unix.AF_NETLINK, unix.SOCK_RAW|unix.SOCK_CLOEXEC, protocol)
	if err != nil {
		return nil, err
	}
	s := &NetlinkSocket{
		fd: int32(fd),
	}
	s.lsa.Family = unix.AF_NETLINK
	if err := unix.Bind(fd, &s.lsa); err != nil {
		unix.Close(fd)
		return nil, err
	}

	return s, nil
}

// GetNetlinkSocketAt opens a netlink socket in the network namespace newNs
// and positions the thread back into the network namespace specified by curNs,
// when done. If curNs is close, the function derives the current namespace and
// moves back into it when done. If newNs is close, the socket will be opened
// in the current network namespace.
func GetNetlinkSocketAt(newNs, curNs netns.NsHandle, protocol int) (*NetlinkSocket, error) {
	c, err := executeInNetns(newNs, curNs)
	if err != nil {
		return nil, err
	}
	defer c()
	return getNetlinkSocket(protocol)
}

// executeInNetns sets execution of the code following this call to the
// network namespace newNs, then moves the thread back to curNs if open,
// otherwise to the current netns at the time the function was invoked
// In case of success, the caller is expected to execute the returned function
// at the end of the code that needs to be executed in the network namespace.
// Example:
// func jobAt(...) error {
//      d, err := executeInNetns(...)
//      if err != nil { return err}
//      defer d()
//      < code which needs to be executed in specific netns>
//  }
// TODO: his function probably belongs to netns pkg.
func executeInNetns(newNs, curNs netns.NsHandle) (func(), error) {
	var (
		err       error
		moveBack  func(netns.NsHandle) error
		closeNs   func() error
		unlockThd func()
	)
	restore := func() {
		// order matters
		if moveBack != nil {
			moveBack(curNs)
		}
		if closeNs != nil {
			closeNs()
		}
		if unlockThd != nil {
			unlockThd()
		}
	}
	if newNs.IsOpen() {
		runtime.LockOSThread()
		unlockThd = runtime.UnlockOSThread
		if !curNs.IsOpen() {
			if curNs, err = netns.Get(); err != nil {
				restore()
				return nil, fmt.Errorf("could not get current namespace while creating netlink socket: %v", err)
			}
			closeNs = curNs.Close
		}
		if err := netns.Set(newNs); err != nil {
			restore()
			return nil, fmt.Errorf("failed to set into network namespace %d while creating netlink socket: %v", newNs, err)
		}
		moveBack = netns.Set
	}
	return restore, nil
}

// Create a netlink socket with a given protocol (e.g. NETLINK_ROUTE)
// and subscribe it to multicast groups passed in variable argument list.
// Returns the netlink socket on which Receive() method can be called
// to retrieve the messages from the kernel.
func Subscribe(protocol int, groups ...uint) (*NetlinkSocket, error) {
	fd, err := unix.Socket(unix.AF_NETLINK, unix.SOCK_RAW, protocol)
	if err != nil {
		return nil, err
	}
	s := &NetlinkSocket{
		fd: int32(fd),
	}
	s.lsa.Family = unix.AF_NETLINK

	for _, g := range groups {
		s.lsa.Groups |= (1 << (g - 1))
	}

	if err := unix.Bind(fd, &s.lsa); err != nil {
		unix.Close(fd)
		return nil, err
	}

	return s, nil
}

// SubscribeAt works like Subscribe plus let's the caller choose the network
// namespace in which the socket would be opened (newNs). Then control goes back
// to curNs if open, otherwise to the netns at the time this function was called.
func SubscribeAt(newNs, curNs netns.NsHandle, protocol int, groups ...uint) (*NetlinkSocket, error) {
	c, err := executeInNetns(newNs, curNs)
	if err != nil {
		return nil, err
	}
	defer c()
	return Subscribe(protocol, groups...)
}

func (s *NetlinkSocket) Close() {
	fd := int(atomic.SwapInt32(&s.fd, -1))
	unix.Close(fd)
}

func (s *NetlinkSocket) GetFd() int {
	return int(atomic.LoadInt32(&s.fd))
}

func (s *NetlinkSocket) Send(request *NetlinkRequest) error {
	fd := int(atomic.LoadInt32(&s.fd))
	if fd < 0 {
		return fmt.Errorf("Send called on a closed socket")
	}
	if err := unix.Sendto(fd, request.Serialize(), 0, &s.lsa); err != nil {
		return err
	}
	return nil
}

func (s *NetlinkSocket) Receive() ([]syscall.NetlinkMessage, *unix.SockaddrNetlink, error) {
	fd := int(atomic.LoadInt32(&s.fd))
	if fd < 0 {
		return nil, nil, fmt.Errorf("Receive called on a closed socket")
	}
	var fromAddr *unix.SockaddrNetlink
	var rb [RECEIVE_BUFFER_SIZE]byte
	nr, from, err := unix.Recvfrom(fd, rb[:], 0)
	if err != nil {
		return nil, nil, err
	}
	fromAddr, ok := from.(*unix.SockaddrNetlink)
	if !ok {
		return nil, nil, fmt.Errorf("Error converting to netlink sockaddr")
	}
	if nr < unix.NLMSG_HDRLEN {
		return nil, nil, fmt.Errorf("Got short response from netlink")
	}
	rb2 := make([]byte, nr)
	copy(rb2, rb[:nr])
	nl, err := syscall.ParseNetlinkMessage(rb2)
	if err != nil {
		return nil, nil, err
	}
	return nl, fromAddr, nil
}

// SetSendTimeout allows to set a send timeout on the socket
func (s *NetlinkSocket) SetSendTimeout(timeout *unix.Timeval) error {
	// Set a send timeout of SOCKET_SEND_TIMEOUT, this will allow the Send to periodically unblock and avoid that a routine
	// remains stuck on a send on a closed fd
	return unix.SetsockoptTimeval(int(s.fd), unix.SOL_SOCKET, unix.SO_SNDTIMEO, timeout)
}

// SetReceiveTimeout allows to set a receive timeout on the socket
func (s *NetlinkSocket) SetReceiveTimeout(timeout *unix.Timeval) error {
	// Set a read timeout of SOCKET_READ_TIMEOUT, this will allow the Read to periodically unblock and avoid that a routine
	// remains stuck on a recvmsg on a closed fd
	return unix.SetsockoptTimeval(int(s.fd), unix.SOL_SOCKET, unix.SO_RCVTIMEO, timeout)
}

func (s *NetlinkSocket) GetPid() (uint32, error) {
	fd := int(atomic.LoadInt32(&s.fd))
	lsa, err := unix.Getsockname(fd)
	if err != nil {
		return 0, err
	}
	switch v := lsa.(type) {
	case *unix.SockaddrNetlink:
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

func Uint64Attr(v uint64) []byte {
	native := NativeEndian()
	bytes := make([]byte, 8)
	native.PutUint64(bytes, v)
	return bytes
}

func ParseRouteAttr(b []byte) ([]syscall.NetlinkRouteAttr, error) {
	var attrs []syscall.NetlinkRouteAttr
	for len(b) >= unix.SizeofRtAttr {
		a, vbuf, alen, err := netlinkRouteAttrAndValue(b)
		if err != nil {
			return nil, err
		}
		ra := syscall.NetlinkRouteAttr{Attr: syscall.RtAttr(*a), Value: vbuf[:int(a.Len)-unix.SizeofRtAttr]}
		attrs = append(attrs, ra)
		b = b[alen:]
	}
	return attrs, nil
}

func netlinkRouteAttrAndValue(b []byte) (*unix.RtAttr, []byte, int, error) {
	a := (*unix.RtAttr)(unsafe.Pointer(&b[0]))
	if int(a.Len) < unix.SizeofRtAttr || int(a.Len) > len(b) {
		return nil, nil, 0, unix.EINVAL
	}
	return a, b[unix.SizeofRtAttr:], rtaAlignOf(int(a.Len)), nil
}

// SocketHandle contains the netlink socket and the associated
// sequence counter for a specific netlink family
type SocketHandle struct {
	Seq    uint32
	Socket *NetlinkSocket
}

// Close closes the netlink socket
func (sh *SocketHandle) Close() {
	if sh.Socket != nil {
		sh.Socket.Close()
	}
}
