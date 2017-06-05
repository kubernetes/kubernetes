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
)

const (
	// Family type definitions
	FAMILY_ALL  = syscall.AF_UNSPEC
	FAMILY_V4   = syscall.AF_INET
	FAMILY_V6   = syscall.AF_INET6
	FAMILY_MPLS = AF_MPLS
)

// SupportedNlFamilies contains the list of netlink families this netlink package supports
var SupportedNlFamilies = []int{syscall.NETLINK_ROUTE, syscall.NETLINK_XFRM, syscall.NETLINK_NETFILTER}

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

func (msg *IfInfomsg) EncapType() string {
	switch msg.Type {
	case 0:
		return "generic"
	case syscall.ARPHRD_ETHER:
		return "ether"
	case syscall.ARPHRD_EETHER:
		return "eether"
	case syscall.ARPHRD_AX25:
		return "ax25"
	case syscall.ARPHRD_PRONET:
		return "pronet"
	case syscall.ARPHRD_CHAOS:
		return "chaos"
	case syscall.ARPHRD_IEEE802:
		return "ieee802"
	case syscall.ARPHRD_ARCNET:
		return "arcnet"
	case syscall.ARPHRD_APPLETLK:
		return "atalk"
	case syscall.ARPHRD_DLCI:
		return "dlci"
	case syscall.ARPHRD_ATM:
		return "atm"
	case syscall.ARPHRD_METRICOM:
		return "metricom"
	case syscall.ARPHRD_IEEE1394:
		return "ieee1394"
	case syscall.ARPHRD_INFINIBAND:
		return "infiniband"
	case syscall.ARPHRD_SLIP:
		return "slip"
	case syscall.ARPHRD_CSLIP:
		return "cslip"
	case syscall.ARPHRD_SLIP6:
		return "slip6"
	case syscall.ARPHRD_CSLIP6:
		return "cslip6"
	case syscall.ARPHRD_RSRVD:
		return "rsrvd"
	case syscall.ARPHRD_ADAPT:
		return "adapt"
	case syscall.ARPHRD_ROSE:
		return "rose"
	case syscall.ARPHRD_X25:
		return "x25"
	case syscall.ARPHRD_HWX25:
		return "hwx25"
	case syscall.ARPHRD_PPP:
		return "ppp"
	case syscall.ARPHRD_HDLC:
		return "hdlc"
	case syscall.ARPHRD_LAPB:
		return "lapb"
	case syscall.ARPHRD_DDCMP:
		return "ddcmp"
	case syscall.ARPHRD_RAWHDLC:
		return "rawhdlc"
	case syscall.ARPHRD_TUNNEL:
		return "ipip"
	case syscall.ARPHRD_TUNNEL6:
		return "tunnel6"
	case syscall.ARPHRD_FRAD:
		return "frad"
	case syscall.ARPHRD_SKIP:
		return "skip"
	case syscall.ARPHRD_LOOPBACK:
		return "loopback"
	case syscall.ARPHRD_LOCALTLK:
		return "ltalk"
	case syscall.ARPHRD_FDDI:
		return "fddi"
	case syscall.ARPHRD_BIF:
		return "bif"
	case syscall.ARPHRD_SIT:
		return "sit"
	case syscall.ARPHRD_IPDDP:
		return "ip/ddp"
	case syscall.ARPHRD_IPGRE:
		return "gre"
	case syscall.ARPHRD_PIMREG:
		return "pimreg"
	case syscall.ARPHRD_HIPPI:
		return "hippi"
	case syscall.ARPHRD_ASH:
		return "ash"
	case syscall.ARPHRD_ECONET:
		return "econet"
	case syscall.ARPHRD_IRDA:
		return "irda"
	case syscall.ARPHRD_FCPP:
		return "fcpp"
	case syscall.ARPHRD_FCAL:
		return "fcal"
	case syscall.ARPHRD_FCPL:
		return "fcpl"
	case syscall.ARPHRD_FCFABRIC:
		return "fcfb0"
	case syscall.ARPHRD_FCFABRIC + 1:
		return "fcfb1"
	case syscall.ARPHRD_FCFABRIC + 2:
		return "fcfb2"
	case syscall.ARPHRD_FCFABRIC + 3:
		return "fcfb3"
	case syscall.ARPHRD_FCFABRIC + 4:
		return "fcfb4"
	case syscall.ARPHRD_FCFABRIC + 5:
		return "fcfb5"
	case syscall.ARPHRD_FCFABRIC + 6:
		return "fcfb6"
	case syscall.ARPHRD_FCFABRIC + 7:
		return "fcfb7"
	case syscall.ARPHRD_FCFABRIC + 8:
		return "fcfb8"
	case syscall.ARPHRD_FCFABRIC + 9:
		return "fcfb9"
	case syscall.ARPHRD_FCFABRIC + 10:
		return "fcfb10"
	case syscall.ARPHRD_FCFABRIC + 11:
		return "fcfb11"
	case syscall.ARPHRD_FCFABRIC + 12:
		return "fcfb12"
	case syscall.ARPHRD_IEEE802_TR:
		return "tr"
	case syscall.ARPHRD_IEEE80211:
		return "ieee802.11"
	case syscall.ARPHRD_IEEE80211_PRISM:
		return "ieee802.11/prism"
	case syscall.ARPHRD_IEEE80211_RADIOTAP:
		return "ieee802.11/radiotap"
	case syscall.ARPHRD_IEEE802154:
		return "ieee802.15.4"

	case 65534:
		return "none"
	case 65535:
		return "void"
	}
	return fmt.Sprintf("unknown%d", msg.Type)
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
	syscall.NlMsghdr
	Data    []NetlinkRequestData
	RawData []byte
	Sockets map[int]*SocketHandle
}

// Serialize the Netlink Request into a byte array
func (req *NetlinkRequest) Serialize() []byte {
	length := syscall.SizeofNlMsghdr
	dataBytes := make([][]byte, len(req.Data))
	for i, data := range req.Data {
		dataBytes[i] = data.Serialize()
		length = length + len(dataBytes[i])
	}
	length += len(req.RawData)

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
	// Add the raw data if any
	if len(req.RawData) > 0 {
		copy(b[next:length], req.RawData)
	}
	return b
}

func (req *NetlinkRequest) AddData(data NetlinkRequestData) {
	if data != nil {
		req.Data = append(req.Data, data)
	}
}

// AddRawData adds raw bytes to the end of the NetlinkRequest object during serialization
func (req *NetlinkRequest) AddRawData(data []byte) {
	if data != nil {
		req.RawData = append(req.RawData, data...)
	}
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
		msgs, err := s.Receive()
		if err != nil {
			return nil, err
		}
		for _, m := range msgs {
			if m.Header.Seq != req.Seq {
				if sharedSocket {
					continue
				}
				return nil, fmt.Errorf("Wrong Seq nr %d, expected %d", m.Header.Seq, req.Seq)
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
	sync.Mutex
}

func getNetlinkSocket(protocol int) (*NetlinkSocket, error) {
	fd, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_RAW|syscall.SOCK_CLOEXEC, protocol)
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
	syscall.Close(s.fd)
	s.fd = -1
}

func (s *NetlinkSocket) GetFd() int {
	return s.fd
}

func (s *NetlinkSocket) Send(request *NetlinkRequest) error {
	if s.fd < 0 {
		return fmt.Errorf("Send called on a closed socket")
	}
	if err := syscall.Sendto(s.fd, request.Serialize(), 0, &s.lsa); err != nil {
		return err
	}
	return nil
}

func (s *NetlinkSocket) Receive() ([]syscall.NetlinkMessage, error) {
	if s.fd < 0 {
		return nil, fmt.Errorf("Receive called on a closed socket")
	}
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

func Uint64Attr(v uint64) []byte {
	native := NativeEndian()
	bytes := make([]byte, 8)
	native.PutUint64(bytes, v)
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
