// +build linux

// Package nlgo implements netlink library routines.
//
// This was golang port of libnl. For basic concept, please have a look at
// original libnl documentation http://www.infradead.org/~tgr/libnl/ .
//
package nlgo

import (
	"fmt"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"
)

func align(size, tick int) int {
	return (size + tick - 1) &^ (tick - 1)
}

func NLMSG_ALIGN(size int) int {
	return align(size, syscall.NLMSG_ALIGNTO)
}

func NLA_ALIGN(size int) int {
	return align(size, syscall.NLA_ALIGNTO)
}

const NLA_TYPE_MASK = ^uint16(syscall.NLA_F_NESTED | syscall.NLA_F_NET_BYTEORDER)

type NlaValue interface {
	// returns the entire nlattr byte sequence including header and trailing alignment padding
	Build(syscall.NlAttr) []byte
}

// Attr represents single netlink attribute package.
// Developer memo: syscall.ParseNetlinkRouteAttr does not parse nested attributes.
// I wanted to make deep parser.
type Attr struct {
	Header syscall.NlAttr
	Value  NlaValue
}

func (self Attr) Field() uint16 {
	return self.Header.Type & NLA_TYPE_MASK
}

func (self Attr) Bytes() []byte {
	return self.Value.Build(self.Header)
}

func (self Attr) Build(hdr syscall.NlAttr) []byte {
	return Binary(self.Bytes()).Build(hdr)
}

func (self Attr) Array() []Attr {
	return []Attr{self}
}

type AttrList interface {
	Slice() []Attr
	Bytes() []byte
}

type AttrSlice []Attr

func (self AttrSlice) Bytes() []byte {
	var ret []byte
	for _, attr := range []Attr(self) {
		ret = append(ret, attr.Value.Build(attr.Header)...)
	}
	return ret
}

func (self AttrSlice) Build(hdr syscall.NlAttr) []byte {
	return Binary(self.Bytes()).Build(hdr)
}

func (self AttrSlice) String() string {
	var comps []string
	for _, attr := range []Attr(self) {
		comps = append(comps, fmt.Sprintf("%d: %v", attr.Field(), attr.Value))
	}
	return fmt.Sprintf("[%s]", strings.Join(comps, ", "))
}

func (self AttrSlice) Slice() []Attr {
	return []Attr(self)
}

type Policy interface {
	Parse([]byte) (NlaValue, error)
}

type SinglePolicy int

const (
	U8Policy SinglePolicy = iota
	U16Policy
	U32Policy
	U64Policy
	StringPolicy
	FlagPolicy
	NulStringPolicy
	BinaryPolicy
	S8Policy
	S16Policy
	S32Policy
	S64Policy
)

func (self SinglePolicy) Parse(nla []byte) (NlaValue, error) {
	if len(nla) < syscall.SizeofNlAttr {
		panic("s1")
		return nil, NLE_RANGE
	}
	hdr := (*syscall.NlAttr)(unsafe.Pointer(&nla[0]))
	if int(hdr.Len) < NLA_HDRLEN || int(hdr.Len) > len(nla) {
		panic("s2")
		return nil, NLE_RANGE
	}
	attr := Attr{
		Header: *hdr,
		Value:  Binary(nla[NLA_HDRLEN:hdr.Len]),
	}
	var err error
	switch self {
	case U8Policy:
		err = setU8(&attr)
	case U16Policy:
		err = setU16(&attr)
	case U32Policy:
		err = setU32(&attr)
	case U64Policy:
		err = setU64(&attr)
	case BinaryPolicy:
		// do nothing
	case StringPolicy:
		err = setString(&attr)
	case NulStringPolicy:
		err = setNulString(&attr)
	case FlagPolicy:
		err = setFlag(&attr)
	case S8Policy:
		err = setS8(&attr)
	case S16Policy:
		err = setS16(&attr)
	case S32Policy:
		err = setS32(&attr)
	case S64Policy:
		err = setS64(&attr)
	default:
		err = fmt.Errorf("unknown single policy")
	}
	return attr, err
}

func parseElement(p Policy, attr *Attr) error {
	if p != nil {
		switch policy := p.(type) {
		case SinglePolicy:
			if value, err := policy.Parse(attr.Bytes()); err != nil {
				return err
			} else {
				*attr = value.(Attr)
			}
		default:
			if value, err := policy.Parse([]byte(attr.Value.(Binary))); err != nil {
				return err
			} else {
				attr.Value = value
			}
		}
	} else if attr.Header.Type&syscall.NLA_F_NESTED != 0 {
		if list, err := SimpleListPolicy.Parse([]byte(attr.Value.(Binary))); err != nil {
			return err
		} else {
			attr.Value = list
		}
	}
	return nil
}

type ListPolicy struct {
	Nested Policy
}

var SimpleListPolicy ListPolicy

func (self ListPolicy) Parse(nla []byte) (NlaValue, error) {
	var attrs []Attr
	for len(nla) >= NLA_HDRLEN {
		var attr Attr
		if sattr, err := BinaryPolicy.Parse(nla); err != nil {
			return nil, err
		} else {
			attr = sattr.(Attr)
		}
		if err := parseElement(self.Nested, &attr); err != nil {
			return nil, err
		}
		attrs = append(attrs, attr)
		nla = nla[NLA_ALIGN(int(attr.Header.Len)):]
	}
	return AttrSlice(attrs), nil
}

type AttrMap struct {
	AttrSlice
	Policy MapPolicy
}

func (self AttrMap) Get(field uint16) NlaValue {
	for _, attr := range self.Slice() {
		if attr.Field() == field {
			return attr.Value
		}
	}
	return nil
}

func (self AttrMap) String() string {
	var comps []string
	for _, attr := range self.AttrSlice {
		field := attr.Field()
		name := "?"
		if n, ok := self.Policy.Names[field]; ok {
			name = n
		}
		comps = append(comps, fmt.Sprintf("%s: %v", name, attr.Value))
	}
	return fmt.Sprintf("%s(%s)", self.Policy.Prefix, strings.Join(comps, ", "))
}

type MapPolicy struct {
	Prefix string
	Names  map[uint16]string
	Rule   map[uint16]Policy
}

func (self MapPolicy) Parse(nla []byte) (NlaValue, error) {
	var attrs []Attr
	for len(nla) >= NLA_HDRLEN {
		var attr Attr
		if sattr, err := BinaryPolicy.Parse(nla); err != nil {
			return nil, err
		} else {
			attr = sattr.(Attr)
		}
		if err := parseElement(self.Rule[attr.Field()], &attr); err != nil {
			return nil, err
		}
		attrs = append(attrs, attr)
		nla = nla[NLA_ALIGN(int(attr.Header.Len)):]
	}
	return AttrMap{
		AttrSlice: attrs,
		Policy:    self,
	}, nil
}

// error.h

type NlError int

const (
	NLE_SUCCESS NlError = iota
	NLE_FAILURE
	NLE_INTR
	NLE_BAD_SOCK
	NLE_AGAIN
	NLE_NOMEM
	NLE_EXIST
	NLE_INVAL
	NLE_RANGE
	NLE_MSGSIZE
	NLE_OPNOTSUPP
	NLE_AF_NOSUPPORT
	NLE_OBJ_NOTFOUND
	NLE_NOATTR
	NLE_MISSING_ATTR
	NLE_AF_MISMATCH
	NLE_SEQ_MISMATCH
	NLE_MSG_OVERFLOW
	NLE_MSG_TRUNC
	NLE_NOADDR
	NLE_SRCRT_NOSUPPORT
	NLE_MSG_TOOSHORT
	NLE_MSGTYPE_NOSUPPORT
	NLE_OBJ_MISMATCH
	NLE_NOCACHE
	NLE_BUSY
	NLE_PROTO_MISMATCH
	NLE_NOACCESS
	NLE_PERM
	NLE_PKTLOC_FILE
	NLE_PARSE_ERR
	NLE_NODEV
	NLE_IMMUTABLE
	NLE_DUMP_INTR
)

func (self NlError) Error() string {
	switch self {
	default:
		return "Unspecific failure"
	case NLE_SUCCESS:
		return "Success"
	case NLE_FAILURE:
		return "Unspecific failure"
	case NLE_INTR:
		return "Interrupted system call"
	case NLE_BAD_SOCK:
		return "Bad socket"
	case NLE_AGAIN:
		return "Try again"
	case NLE_NOMEM:
		return "Out of memory"
	case NLE_EXIST:
		return "Object exists"
	case NLE_INVAL:
		return "Invalid input data or parameter"
	case NLE_RANGE:
		return "Input data out of range"
	case NLE_MSGSIZE:
		return "Message size not sufficient"
	case NLE_OPNOTSUPP:
		return "Operation not supported"
	case NLE_AF_NOSUPPORT:
		return "Address family not supported"
	case NLE_OBJ_NOTFOUND:
		return "Object not found"
	case NLE_NOATTR:
		return "Attribute not available"
	case NLE_MISSING_ATTR:
		return "Missing attribute"
	case NLE_AF_MISMATCH:
		return "Address family mismatch"
	case NLE_SEQ_MISMATCH:
		return "Message sequence number mismatch"
	case NLE_MSG_OVERFLOW:
		return "Kernel reported message overflow"
	case NLE_MSG_TRUNC:
		return "Kernel reported truncated message"
	case NLE_NOADDR:
		return "Invalid address for specified address family"
	case NLE_SRCRT_NOSUPPORT:
		return "Source based routing not supported"
	case NLE_MSG_TOOSHORT:
		return "Netlink message is too short"
	case NLE_MSGTYPE_NOSUPPORT:
		return "Netlink message type is not supported"
	case NLE_OBJ_MISMATCH:
		return "Object type does not match cache"
	case NLE_NOCACHE:
		return "Unknown or invalid cache type"
	case NLE_BUSY:
		return "Object busy"
	case NLE_PROTO_MISMATCH:
		return "Protocol mismatch"
	case NLE_NOACCESS:
		return "No Access"
	case NLE_PERM:
		return "Operation not permitted"
	case NLE_PKTLOC_FILE:
		return "Unable to open packet location file"
	case NLE_PARSE_ERR:
		return "Unable to parse object"
	case NLE_NODEV:
		return "No such device"
	case NLE_IMMUTABLE:
		return "Immutable attribute"
	case NLE_DUMP_INTR:
		return "Dump inconsistency detected, interrupted"
	}
}

// socket.c

var pidLock = &sync.Mutex{}
var pidUsed = make(map[int]bool)

const (
	NL_SOCK_BUFSIZE_SET = 1 << iota
	NL_SOCK_PASSCRED
	NL_OWN_PORT
	NL_MSG_PEEK
	NL_NO_AUTO_ACK
)

type NlSock struct {
	Local     syscall.SockaddrNetlink
	Peer      syscall.SockaddrNetlink
	Fd        int
	SeqNext   uint32
	SeqExpect uint32
	Flags     int // NL_NO_AUTO_ACK etc.,
}

func NlSocketAlloc() *NlSock {
	tick := uint32(time.Now().Unix())
	return &NlSock{
		Fd: -1,
		Local: syscall.SockaddrNetlink{
			Family: syscall.AF_NETLINK,
		},
		Peer: syscall.SockaddrNetlink{
			Family: syscall.AF_NETLINK,
		},
		SeqNext:   tick,
		SeqExpect: tick,
		Flags:     NL_OWN_PORT,
	}
}

func NlSocketFree(sk *NlSock) {
	if sk.Fd >= 0 {
		syscall.Close(sk.Fd)
	}
	pidLock.Lock()
	defer func() {
		pidLock.Unlock()
	}()
	high := sk.Local.Pid >> 22
	delete(pidUsed, int(high))
}

func NlSocketSetBufferSize(sk *NlSock, rxbuf, txbuf int) error {
	if rxbuf <= 0 {
		rxbuf = 32768
	}
	if txbuf <= 0 {
		txbuf = 32768
	}
	if sk.Fd == -1 {
		return NLE_BAD_SOCK
	}
	if err := syscall.SetsockoptInt(sk.Fd, syscall.SOL_SOCKET, syscall.SO_SNDBUF, txbuf); err != nil {
		return err
	}
	if err := syscall.SetsockoptInt(sk.Fd, syscall.SOL_SOCKET, syscall.SO_RCVBUF, rxbuf); err != nil {
		return err
	}
	sk.Flags |= NL_SOCK_BUFSIZE_SET
	return nil
}

// NlConnect is same with libnl nl_connect. nl_close is required for releaseing internal fd.
func NlConnect(sk *NlSock, protocol int) error {
	if sk.Fd != -1 {
		return NLE_BAD_SOCK
	}
	if fd, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_RAW|syscall.SOCK_CLOEXEC, protocol); err != nil {
		return err
	} else {
		sk.Fd = fd
	}
	if sk.Flags&NL_SOCK_BUFSIZE_SET != 0 {
		if err := NlSocketSetBufferSize(sk, 0, 0); err != nil {
			return err
		}
	}
	if sk.Local.Pid == 0 { // _nl_socket_is_local_port_unspecified
		// kernel will assign Pid
		local := &syscall.SockaddrNetlink{
			Family: syscall.AF_NETLINK,
		}
		if err := syscall.Bind(sk.Fd, local); err != nil {
			return err
		} else if sa, err := syscall.Getsockname(sk.Fd); err != nil {
			return err
		} else {
			local = sa.(*syscall.SockaddrNetlink)
		}
		if local == nil {
			return NLE_EXIST
		}
		sk.Local = *local
		sk.Flags &^= NL_OWN_PORT
	} else {
		if err := syscall.Bind(sk.Fd, &(sk.Local)); err != nil {
			return err
		}
	}
	return nil
}

// NlSocketAddMembership is same with libnl nl_socket_add_membership.
func NlSocketAddMembership(sk *NlSock, group int) error {
	return syscall.SetsockoptInt(sk.Fd, SOL_NETLINK, syscall.NETLINK_ADD_MEMBERSHIP, group)
}

// NlSocketAddMembership is same with libnl nl_socket_drop_membership.
func NlSocketDropMembership(sk *NlSock, group int) error {
	return syscall.SetsockoptInt(sk.Fd, SOL_NETLINK, syscall.NETLINK_DROP_MEMBERSHIP, group)
}

// msg.c

const NL_AUTO_PORT = 0
const NL_AUTO_SEQ = 0

// NlSendSimple is same with libnl nl_send_simple.
func NlSendSimple(sk *NlSock, family uint16, flags uint16, buf []byte) error {
	msg := make([]byte, syscall.NLMSG_HDRLEN+NLMSG_ALIGN(len(buf)))
	hdr := (*syscall.NlMsghdr)(unsafe.Pointer(&msg[0]))
	hdr.Type = family
	hdr.Flags = flags
	hdr.Len = syscall.NLMSG_HDRLEN + uint32(len(buf))
	copy(msg[syscall.NLMSG_HDRLEN:], buf)
	NlCompleteMsg(sk, msg)
	return syscall.Sendto(sk.Fd, msg, 0, &sk.Peer)
}

// nl.c

// NlCompleteMsg is same with libnl nl_complete_msg.
func NlCompleteMsg(sk *NlSock, msg []byte) {
	hdr := (*syscall.NlMsghdr)(unsafe.Pointer(&msg[0]))
	if hdr.Pid == NL_AUTO_PORT {
		hdr.Pid = sk.Local.Pid
	}
	if hdr.Seq == NL_AUTO_SEQ {
		hdr.Seq = sk.SeqNext
		sk.SeqNext++
	}
	hdr.Flags |= syscall.NLM_F_REQUEST
	if sk.Flags&NL_NO_AUTO_ACK == 0 {
		hdr.Flags |= syscall.NLM_F_ACK
	}
}
