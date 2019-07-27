package sctp

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

const (
	SOL_SCTP = 132

	SCTP_BINDX_ADD_ADDR = 0x01
	SCTP_BINDX_REM_ADDR = 0x02

	MSG_NOTIFICATION = 0x8000
)

const (
	SCTP_RTOINFO = iota
	SCTP_ASSOCINFO
	SCTP_INITMSG
	SCTP_NODELAY
	SCTP_AUTOCLOSE
	SCTP_SET_PEER_PRIMARY_ADDR
	SCTP_PRIMARY_ADDR
	SCTP_ADAPTATION_LAYER
	SCTP_DISABLE_FRAGMENTS
	SCTP_PEER_ADDR_PARAMS
	SCTP_DEFAULT_SENT_PARAM
	SCTP_EVENTS
	SCTP_I_WANT_MAPPED_V4_ADDR
	SCTP_MAXSEG
	SCTP_STATUS
	SCTP_GET_PEER_ADDR_INFO
	SCTP_DELAYED_ACK_TIME
	SCTP_DELAYED_ACK  = SCTP_DELAYED_ACK_TIME
	SCTP_DELAYED_SACK = SCTP_DELAYED_ACK_TIME

	SCTP_SOCKOPT_BINDX_ADD = 100
	SCTP_SOCKOPT_BINDX_REM = 101
	SCTP_SOCKOPT_PEELOFF   = 102
	SCTP_GET_PEER_ADDRS    = 108
	SCTP_GET_LOCAL_ADDRS   = 109
	SCTP_SOCKOPT_CONNECTX  = 110
	SCTP_SOCKOPT_CONNECTX3 = 111
)

const (
	SCTP_EVENT_DATA_IO = 1 << iota
	SCTP_EVENT_ASSOCIATION
	SCTP_EVENT_ADDRESS
	SCTP_EVENT_SEND_FAILURE
	SCTP_EVENT_PEER_ERROR
	SCTP_EVENT_SHUTDOWN
	SCTP_EVENT_PARTIAL_DELIVERY
	SCTP_EVENT_ADAPTATION_LAYER
	SCTP_EVENT_AUTHENTICATION
	SCTP_EVENT_SENDER_DRY

	SCTP_EVENT_ALL = SCTP_EVENT_DATA_IO | SCTP_EVENT_ASSOCIATION | SCTP_EVENT_ADDRESS | SCTP_EVENT_SEND_FAILURE | SCTP_EVENT_PEER_ERROR | SCTP_EVENT_SHUTDOWN | SCTP_EVENT_PARTIAL_DELIVERY | SCTP_EVENT_ADAPTATION_LAYER | SCTP_EVENT_AUTHENTICATION | SCTP_EVENT_SENDER_DRY
)

type SCTPNotificationType int

const (
	SCTP_SN_TYPE_BASE = SCTPNotificationType(iota + (1 << 15))
	SCTP_ASSOC_CHANGE
	SCTP_PEER_ADDR_CHANGE
	SCTP_SEND_FAILED
	SCTP_REMOTE_ERROR
	SCTP_SHUTDOWN_EVENT
	SCTP_PARTIAL_DELIVERY_EVENT
	SCTP_ADAPTATION_INDICATION
	SCTP_AUTHENTICATION_INDICATION
	SCTP_SENDER_DRY_EVENT
)

type NotificationHandler func([]byte) error

type EventSubscribe struct {
	DataIO          uint8
	Association     uint8
	Address         uint8
	SendFailure     uint8
	PeerError       uint8
	Shutdown        uint8
	PartialDelivery uint8
	AdaptationLayer uint8
	Authentication  uint8
	SenderDry       uint8
}

const (
	SCTP_CMSG_INIT = iota
	SCTP_CMSG_SNDRCV
	SCTP_CMSG_SNDINFO
	SCTP_CMSG_RCVINFO
	SCTP_CMSG_NXTINFO
)

const (
	SCTP_UNORDERED = 1 << iota
	SCTP_ADDR_OVER
	SCTP_ABORT
	SCTP_SACK_IMMEDIATELY
	SCTP_EOF
)

const (
	SCTP_MAX_STREAM = 0xffff
)

type InitMsg struct {
	NumOstreams    uint16
	MaxInstreams   uint16
	MaxAttempts    uint16
	MaxInitTimeout uint16
}

type SndRcvInfo struct {
	Stream  uint16
	SSN     uint16
	Flags   uint16
	_       uint16
	PPID    uint32
	Context uint32
	TTL     uint32
	TSN     uint32
	CumTSN  uint32
	AssocID int32
}

type SndInfo struct {
	SID     uint16
	Flags   uint16
	PPID    uint32
	Context uint32
	AssocID int32
}

type GetAddrsOld struct {
	AssocID int32
	AddrNum int32
	Addrs   uintptr
}

type NotificationHeader struct {
	Type   uint16
	Flags  uint16
	Length uint32
}

type SCTPState uint16

const (
	SCTP_COMM_UP = SCTPState(iota)
	SCTP_COMM_LOST
	SCTP_RESTART
	SCTP_SHUTDOWN_COMP
	SCTP_CANT_STR_ASSOC
)

var nativeEndian binary.ByteOrder
var sndRcvInfoSize uintptr

func init() {
	i := uint16(1)
	if *(*byte)(unsafe.Pointer(&i)) == 0 {
		nativeEndian = binary.BigEndian
	} else {
		nativeEndian = binary.LittleEndian
	}
	info := SndRcvInfo{}
	sndRcvInfoSize = unsafe.Sizeof(info)
}

func toBuf(v interface{}) []byte {
	var buf bytes.Buffer
	binary.Write(&buf, nativeEndian, v)
	return buf.Bytes()
}

func htons(h uint16) uint16 {
	if nativeEndian == binary.LittleEndian {
		return (h << 8 & 0xff00) | (h >> 8 & 0xff)
	}
	return h
}

var ntohs = htons

// setInitOpts sets options for an SCTP association initialization
// see https://tools.ietf.org/html/rfc4960#page-25
func setInitOpts(fd int, options InitMsg) error {
	optlen := unsafe.Sizeof(options)
	_, _, err := setsockopt(fd, SCTP_INITMSG, uintptr(unsafe.Pointer(&options)), uintptr(optlen))
	return err
}

func setNumOstreams(fd, num int) error {
	return setInitOpts(fd, InitMsg{NumOstreams: uint16(num)})
}

type SCTPAddr struct {
	IPAddrs []net.IPAddr
	Port    int
}

func (a *SCTPAddr) ToRawSockAddrBuf() []byte {
	p := htons(uint16(a.Port))
	if len(a.IPAddrs) == 0 { // if a.IPAddrs list is empty - fall back to IPv4 zero addr
		s := syscall.RawSockaddrInet4{
			Family: syscall.AF_INET,
			Port:   p,
		}
		copy(s.Addr[:], net.IPv4zero)
		return toBuf(s)
	}
	buf := []byte{}
	for _, ip := range a.IPAddrs {
		ipBytes := ip.IP
		if len(ipBytes) == 0 {
			ipBytes = net.IPv4zero
		}
		if ip4 := ipBytes.To4(); ip4 != nil {
			s := syscall.RawSockaddrInet4{
				Family: syscall.AF_INET,
				Port:   p,
			}
			copy(s.Addr[:], ip4)
			buf = append(buf, toBuf(s)...)
		} else {
			var scopeid uint32
			ifi, err := net.InterfaceByName(ip.Zone)
			if err == nil {
				scopeid = uint32(ifi.Index)
			}
			s := syscall.RawSockaddrInet6{
				Family:   syscall.AF_INET6,
				Port:     p,
				Scope_id: scopeid,
			}
			copy(s.Addr[:], ipBytes)
			buf = append(buf, toBuf(s)...)
		}
	}
	return buf
}

func (a *SCTPAddr) String() string {
	var b bytes.Buffer

	for n, i := range a.IPAddrs {
		if i.IP.To4() != nil {
			b.WriteString(i.String())
		} else if i.IP.To16() != nil {
			b.WriteRune('[')
			b.WriteString(i.String())
			b.WriteRune(']')
		}
		if n < len(a.IPAddrs)-1 {
			b.WriteRune('/')
		}
	}
	b.WriteRune(':')
	b.WriteString(strconv.Itoa(a.Port))
	return b.String()
}

func (a *SCTPAddr) Network() string { return "sctp" }

func ResolveSCTPAddr(network, addrs string) (*SCTPAddr, error) {
	tcpnet := ""
	switch network {
	case "", "sctp":
		tcpnet = "tcp"
	case "sctp4":
		tcpnet = "tcp4"
	case "sctp6":
		tcpnet = "tcp6"
	default:
		return nil, fmt.Errorf("invalid net: %s", network)
	}
	elems := strings.Split(addrs, "/")
	if len(elems) == 0 {
		return nil, fmt.Errorf("invalid input: %s", addrs)
	}
	ipaddrs := make([]net.IPAddr, 0, len(elems))
	for _, e := range elems[:len(elems)-1] {
		tcpa, err := net.ResolveTCPAddr(tcpnet, e+":")
		if err != nil {
			return nil, err
		}
		ipaddrs = append(ipaddrs, net.IPAddr{IP: tcpa.IP, Zone: tcpa.Zone})
	}
	tcpa, err := net.ResolveTCPAddr(tcpnet, elems[len(elems)-1])
	if err != nil {
		return nil, err
	}
	if tcpa.IP != nil {
		ipaddrs = append(ipaddrs, net.IPAddr{IP: tcpa.IP, Zone: tcpa.Zone})
	} else {
		ipaddrs = nil
	}
	return &SCTPAddr{
		IPAddrs: ipaddrs,
		Port:    tcpa.Port,
	}, nil
}

func SCTPConnect(fd int, addr *SCTPAddr) (int, error) {
	buf := addr.ToRawSockAddrBuf()
	param := GetAddrsOld{
		AddrNum: int32(len(buf)),
		Addrs:   uintptr(uintptr(unsafe.Pointer(&buf[0]))),
	}
	optlen := unsafe.Sizeof(param)
	_, _, err := getsockopt(fd, SCTP_SOCKOPT_CONNECTX3, uintptr(unsafe.Pointer(&param)), uintptr(unsafe.Pointer(&optlen)))
	if err == nil {
		return int(param.AssocID), nil
	} else if err != syscall.ENOPROTOOPT {
		return 0, err
	}
	r0, _, err := setsockopt(fd, SCTP_SOCKOPT_CONNECTX, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	return int(r0), err
}

func SCTPBind(fd int, addr *SCTPAddr, flags int) error {
	var option uintptr
	switch flags {
	case SCTP_BINDX_ADD_ADDR:
		option = SCTP_SOCKOPT_BINDX_ADD
	case SCTP_BINDX_REM_ADDR:
		option = SCTP_SOCKOPT_BINDX_REM
	default:
		return syscall.EINVAL
	}

	buf := addr.ToRawSockAddrBuf()
	_, _, err := setsockopt(fd, option, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	return err
}

type SCTPConn struct {
	_fd                 int32
	notificationHandler NotificationHandler
}

func (c *SCTPConn) fd() int {
	return int(atomic.LoadInt32(&c._fd))
}

func NewSCTPConn(fd int, handler NotificationHandler) *SCTPConn {
	conn := &SCTPConn{
		_fd:                 int32(fd),
		notificationHandler: handler,
	}
	return conn
}

func (c *SCTPConn) Write(b []byte) (int, error) {
	return c.SCTPWrite(b, nil)
}

func (c *SCTPConn) Read(b []byte) (int, error) {
	n, _, err := c.SCTPRead(b)
	if n < 0 {
		n = 0
	}
	return n, err
}

func (c *SCTPConn) SetInitMsg(numOstreams, maxInstreams, maxAttempts, maxInitTimeout int) error {
	return setInitOpts(c.fd(), InitMsg{
		NumOstreams:    uint16(numOstreams),
		MaxInstreams:   uint16(maxInstreams),
		MaxAttempts:    uint16(maxAttempts),
		MaxInitTimeout: uint16(maxInitTimeout),
	})
}

func (c *SCTPConn) SubscribeEvents(flags int) error {
	var d, a, ad, sf, p, sh, pa, ada, au, se uint8
	if flags&SCTP_EVENT_DATA_IO > 0 {
		d = 1
	}
	if flags&SCTP_EVENT_ASSOCIATION > 0 {
		a = 1
	}
	if flags&SCTP_EVENT_ADDRESS > 0 {
		ad = 1
	}
	if flags&SCTP_EVENT_SEND_FAILURE > 0 {
		sf = 1
	}
	if flags&SCTP_EVENT_PEER_ERROR > 0 {
		p = 1
	}
	if flags&SCTP_EVENT_SHUTDOWN > 0 {
		sh = 1
	}
	if flags&SCTP_EVENT_PARTIAL_DELIVERY > 0 {
		pa = 1
	}
	if flags&SCTP_EVENT_ADAPTATION_LAYER > 0 {
		ada = 1
	}
	if flags&SCTP_EVENT_AUTHENTICATION > 0 {
		au = 1
	}
	if flags&SCTP_EVENT_SENDER_DRY > 0 {
		se = 1
	}
	param := EventSubscribe{
		DataIO:          d,
		Association:     a,
		Address:         ad,
		SendFailure:     sf,
		PeerError:       p,
		Shutdown:        sh,
		PartialDelivery: pa,
		AdaptationLayer: ada,
		Authentication:  au,
		SenderDry:       se,
	}
	optlen := unsafe.Sizeof(param)
	_, _, err := setsockopt(c.fd(), SCTP_EVENTS, uintptr(unsafe.Pointer(&param)), uintptr(optlen))
	return err
}

func (c *SCTPConn) SubscribedEvents() (int, error) {
	param := EventSubscribe{}
	optlen := unsafe.Sizeof(param)
	_, _, err := getsockopt(c.fd(), SCTP_EVENTS, uintptr(unsafe.Pointer(&param)), uintptr(unsafe.Pointer(&optlen)))
	if err != nil {
		return 0, err
	}
	var flags int
	if param.DataIO > 0 {
		flags |= SCTP_EVENT_DATA_IO
	}
	if param.Association > 0 {
		flags |= SCTP_EVENT_ASSOCIATION
	}
	if param.Address > 0 {
		flags |= SCTP_EVENT_ADDRESS
	}
	if param.SendFailure > 0 {
		flags |= SCTP_EVENT_SEND_FAILURE
	}
	if param.PeerError > 0 {
		flags |= SCTP_EVENT_PEER_ERROR
	}
	if param.Shutdown > 0 {
		flags |= SCTP_EVENT_SHUTDOWN
	}
	if param.PartialDelivery > 0 {
		flags |= SCTP_EVENT_PARTIAL_DELIVERY
	}
	if param.AdaptationLayer > 0 {
		flags |= SCTP_EVENT_ADAPTATION_LAYER
	}
	if param.Authentication > 0 {
		flags |= SCTP_EVENT_AUTHENTICATION
	}
	if param.SenderDry > 0 {
		flags |= SCTP_EVENT_SENDER_DRY
	}
	return flags, nil
}

func (c *SCTPConn) SetDefaultSentParam(info *SndRcvInfo) error {
	optlen := unsafe.Sizeof(*info)
	_, _, err := setsockopt(c.fd(), SCTP_DEFAULT_SENT_PARAM, uintptr(unsafe.Pointer(info)), uintptr(optlen))
	return err
}

func (c *SCTPConn) GetDefaultSentParam() (*SndRcvInfo, error) {
	info := &SndRcvInfo{}
	optlen := unsafe.Sizeof(*info)
	_, _, err := getsockopt(c.fd(), SCTP_DEFAULT_SENT_PARAM, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(&optlen)))
	return info, err
}

func resolveFromRawAddr(ptr unsafe.Pointer, n int) (*SCTPAddr, error) {
	addr := &SCTPAddr{
		IPAddrs: make([]net.IPAddr, n),
	}

	switch family := (*(*syscall.RawSockaddrAny)(ptr)).Addr.Family; family {
	case syscall.AF_INET:
		addr.Port = int(ntohs(uint16((*(*syscall.RawSockaddrInet4)(ptr)).Port)))
		tmp := syscall.RawSockaddrInet4{}
		size := unsafe.Sizeof(tmp)
		for i := 0; i < n; i++ {
			a := *(*syscall.RawSockaddrInet4)(unsafe.Pointer(
				uintptr(ptr) + size*uintptr(i)))
			addr.IPAddrs[i] = net.IPAddr{IP: a.Addr[:]}
		}
	case syscall.AF_INET6:
		addr.Port = int(ntohs(uint16((*(*syscall.RawSockaddrInet4)(ptr)).Port)))
		tmp := syscall.RawSockaddrInet6{}
		size := unsafe.Sizeof(tmp)
		for i := 0; i < n; i++ {
			a := *(*syscall.RawSockaddrInet6)(unsafe.Pointer(
				uintptr(ptr) + size*uintptr(i)))
			var zone string
			ifi, err := net.InterfaceByIndex(int(a.Scope_id))
			if err == nil {
				zone = ifi.Name
			}
			addr.IPAddrs[i] = net.IPAddr{IP: a.Addr[:], Zone: zone}
		}
	default:
		return nil, fmt.Errorf("unknown address family: %d", family)
	}
	return addr, nil
}

func sctpGetAddrs(fd, id, optname int) (*SCTPAddr, error) {

	type getaddrs struct {
		assocId int32
		addrNum uint32
		addrs   [4096]byte
	}
	param := getaddrs{
		assocId: int32(id),
	}
	optlen := unsafe.Sizeof(param)
	_, _, err := getsockopt(fd, uintptr(optname), uintptr(unsafe.Pointer(&param)), uintptr(unsafe.Pointer(&optlen)))
	if err != nil {
		return nil, err
	}
	return resolveFromRawAddr(unsafe.Pointer(&param.addrs), int(param.addrNum))
}

func (c *SCTPConn) SCTPGetPrimaryPeerAddr() (*SCTPAddr, error) {

	type sctpGetSetPrim struct {
		assocId int32
		addrs   [128]byte
	}
	param := sctpGetSetPrim{
		assocId: int32(0),
	}
	optlen := unsafe.Sizeof(param)
	_, _, err := getsockopt(c.fd(), SCTP_PRIMARY_ADDR, uintptr(unsafe.Pointer(&param)), uintptr(unsafe.Pointer(&optlen)))
	if err != nil {
		return nil, err
	}
	return resolveFromRawAddr(unsafe.Pointer(&param.addrs), 1)
}

func (c *SCTPConn) SCTPLocalAddr(id int) (*SCTPAddr, error) {
	return sctpGetAddrs(c.fd(), id, SCTP_GET_LOCAL_ADDRS)
}

func (c *SCTPConn) SCTPRemoteAddr(id int) (*SCTPAddr, error) {
	return sctpGetAddrs(c.fd(), id, SCTP_GET_PEER_ADDRS)
}

func (c *SCTPConn) LocalAddr() net.Addr {
	addr, err := sctpGetAddrs(c.fd(), 0, SCTP_GET_LOCAL_ADDRS)
	if err != nil {
		return nil
	}
	return addr
}

func (c *SCTPConn) RemoteAddr() net.Addr {
	addr, err := sctpGetAddrs(c.fd(), 0, SCTP_GET_PEER_ADDRS)
	if err != nil {
		return nil
	}
	return addr
}

func (c *SCTPConn) PeelOff(id int) (*SCTPConn, error) {
	type peeloffArg struct {
		assocId int32
		sd      int
	}
	param := peeloffArg{
		assocId: int32(id),
	}
	optlen := unsafe.Sizeof(param)
	_, _, err := getsockopt(c.fd(), SCTP_SOCKOPT_PEELOFF, uintptr(unsafe.Pointer(&param)), uintptr(unsafe.Pointer(&optlen)))
	if err != nil {
		return nil, err
	}
	return &SCTPConn{_fd: int32(param.sd)}, nil
}

func (c *SCTPConn) SetDeadline(t time.Time) error {
	return syscall.EOPNOTSUPP
}

func (c *SCTPConn) SetReadDeadline(t time.Time) error {
	return syscall.EOPNOTSUPP
}

func (c *SCTPConn) SetWriteDeadline(t time.Time) error {
	return syscall.EOPNOTSUPP
}

type SCTPListener struct {
	fd int
	m  sync.Mutex
}

func (ln *SCTPListener) Addr() net.Addr {
	laddr, err := sctpGetAddrs(ln.fd, 0, SCTP_GET_LOCAL_ADDRS)
	if err != nil {
		return nil
	}
	return laddr
}

type SCTPSndRcvInfoWrappedConn struct {
	conn *SCTPConn
}

func NewSCTPSndRcvInfoWrappedConn(conn *SCTPConn) *SCTPSndRcvInfoWrappedConn {
	conn.SubscribeEvents(SCTP_EVENT_DATA_IO)
	return &SCTPSndRcvInfoWrappedConn{conn}
}

func (c *SCTPSndRcvInfoWrappedConn) Write(b []byte) (int, error) {
	if len(b) < int(sndRcvInfoSize) {
		return 0, syscall.EINVAL
	}
	info := (*SndRcvInfo)(unsafe.Pointer(&b[0]))
	n, err := c.conn.SCTPWrite(b[sndRcvInfoSize:], info)
	return n + int(sndRcvInfoSize), err
}

func (c *SCTPSndRcvInfoWrappedConn) Read(b []byte) (int, error) {
	if len(b) < int(sndRcvInfoSize) {
		return 0, syscall.EINVAL
	}
	n, info, err := c.conn.SCTPRead(b[sndRcvInfoSize:])
	if err != nil {
		return n, err
	}
	copy(b, toBuf(info))
	return n + int(sndRcvInfoSize), err
}

func (c *SCTPSndRcvInfoWrappedConn) Close() error {
	return c.conn.Close()
}

func (c *SCTPSndRcvInfoWrappedConn) LocalAddr() net.Addr {
	return c.conn.LocalAddr()
}

func (c *SCTPSndRcvInfoWrappedConn) RemoteAddr() net.Addr {
	return c.conn.RemoteAddr()
}

func (c *SCTPSndRcvInfoWrappedConn) SetDeadline(t time.Time) error {
	return c.conn.SetDeadline(t)
}

func (c *SCTPSndRcvInfoWrappedConn) SetReadDeadline(t time.Time) error {
	return c.conn.SetReadDeadline(t)
}

func (c *SCTPSndRcvInfoWrappedConn) SetWriteDeadline(t time.Time) error {
	return c.conn.SetWriteDeadline(t)
}

func (c *SCTPSndRcvInfoWrappedConn) SetWriteBuffer(bytes int) error {
	return c.conn.SetWriteBuffer(bytes)
}

func (c *SCTPSndRcvInfoWrappedConn) GetWriteBuffer() (int, error) {
	return c.conn.GetWriteBuffer()
}

func (c *SCTPSndRcvInfoWrappedConn) SetReadBuffer(bytes int) error {
	return c.conn.SetReadBuffer(bytes)
}

func (c *SCTPSndRcvInfoWrappedConn) GetReadBuffer() (int, error) {
	return c.conn.GetReadBuffer()
}
