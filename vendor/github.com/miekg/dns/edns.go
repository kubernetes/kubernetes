package dns

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"net"
	"strconv"
)

// EDNS0 Option codes.
const (
	EDNS0LLQ          = 0x1     // long lived queries: http://tools.ietf.org/html/draft-sekar-dns-llq-01
	EDNS0UL           = 0x2     // update lease draft: http://files.dns-sd.org/draft-sekar-dns-ul.txt
	EDNS0NSID         = 0x3     // nsid (See RFC 5001)
	EDNS0DAU          = 0x5     // DNSSEC Algorithm Understood
	EDNS0DHU          = 0x6     // DS Hash Understood
	EDNS0N3U          = 0x7     // NSEC3 Hash Understood
	EDNS0SUBNET       = 0x8     // client-subnet (See RFC 7871)
	EDNS0EXPIRE       = 0x9     // EDNS0 expire
	EDNS0COOKIE       = 0xa     // EDNS0 Cookie
	EDNS0TCPKEEPALIVE = 0xb     // EDNS0 tcp keep alive (See RFC 7828)
	EDNS0PADDING      = 0xc     // EDNS0 padding (See RFC 7830)
	EDNS0LOCALSTART   = 0xFDE9  // Beginning of range reserved for local/experimental use (See RFC 6891)
	EDNS0LOCALEND     = 0xFFFE  // End of range reserved for local/experimental use (See RFC 6891)
	_DO               = 1 << 15 // DNSSEC OK
)

// OPT is the EDNS0 RR appended to messages to convey extra (meta) information.
// See RFC 6891.
type OPT struct {
	Hdr    RR_Header
	Option []EDNS0 `dns:"opt"`
}

func (rr *OPT) String() string {
	s := "\n;; OPT PSEUDOSECTION:\n; EDNS: version " + strconv.Itoa(int(rr.Version())) + "; "
	if rr.Do() {
		s += "flags: do; "
	} else {
		s += "flags: ; "
	}
	s += "udp: " + strconv.Itoa(int(rr.UDPSize()))

	for _, o := range rr.Option {
		switch o.(type) {
		case *EDNS0_NSID:
			s += "\n; NSID: " + o.String()
			h, e := o.pack()
			var r string
			if e == nil {
				for _, c := range h {
					r += "(" + string(c) + ")"
				}
				s += "  " + r
			}
		case *EDNS0_SUBNET:
			s += "\n; SUBNET: " + o.String()
		case *EDNS0_COOKIE:
			s += "\n; COOKIE: " + o.String()
		case *EDNS0_UL:
			s += "\n; UPDATE LEASE: " + o.String()
		case *EDNS0_LLQ:
			s += "\n; LONG LIVED QUERIES: " + o.String()
		case *EDNS0_DAU:
			s += "\n; DNSSEC ALGORITHM UNDERSTOOD: " + o.String()
		case *EDNS0_DHU:
			s += "\n; DS HASH UNDERSTOOD: " + o.String()
		case *EDNS0_N3U:
			s += "\n; NSEC3 HASH UNDERSTOOD: " + o.String()
		case *EDNS0_LOCAL:
			s += "\n; LOCAL OPT: " + o.String()
		case *EDNS0_PADDING:
			s += "\n; PADDING: " + o.String()
		}
	}
	return s
}

func (rr *OPT) len(off int, compression map[string]struct{}) int {
	l := rr.Hdr.len(off, compression)
	for _, o := range rr.Option {
		l += 4 // Account for 2-byte option code and 2-byte option length.
		lo, _ := o.pack()
		l += len(lo)
	}
	return l
}

func (rr *OPT) parse(c *zlexer, origin string) *ParseError {
	panic("dns: internal error: parse should never be called on OPT")
}

func (r1 *OPT) isDuplicate(r2 RR) bool { return false }

// return the old value -> delete SetVersion?

// Version returns the EDNS version used. Only zero is defined.
func (rr *OPT) Version() uint8 {
	return uint8(rr.Hdr.Ttl & 0x00FF0000 >> 16)
}

// SetVersion sets the version of EDNS. This is usually zero.
func (rr *OPT) SetVersion(v uint8) {
	rr.Hdr.Ttl = rr.Hdr.Ttl&0xFF00FFFF | uint32(v)<<16
}

// ExtendedRcode returns the EDNS extended RCODE field (the upper 8 bits of the TTL).
func (rr *OPT) ExtendedRcode() int {
	return int(rr.Hdr.Ttl&0xFF000000>>24) << 4
}

// SetExtendedRcode sets the EDNS extended RCODE field.
//
// If the RCODE is not an extended RCODE, will reset the extended RCODE field to 0.
func (rr *OPT) SetExtendedRcode(v uint16) {
	rr.Hdr.Ttl = rr.Hdr.Ttl&0x00FFFFFF | uint32(v>>4)<<24
}

// UDPSize returns the UDP buffer size.
func (rr *OPT) UDPSize() uint16 {
	return rr.Hdr.Class
}

// SetUDPSize sets the UDP buffer size.
func (rr *OPT) SetUDPSize(size uint16) {
	rr.Hdr.Class = size
}

// Do returns the value of the DO (DNSSEC OK) bit.
func (rr *OPT) Do() bool {
	return rr.Hdr.Ttl&_DO == _DO
}

// SetDo sets the DO (DNSSEC OK) bit.
// If we pass an argument, set the DO bit to that value.
// It is possible to pass 2 or more arguments. Any arguments after the 1st is silently ignored.
func (rr *OPT) SetDo(do ...bool) {
	if len(do) == 1 {
		if do[0] {
			rr.Hdr.Ttl |= _DO
		} else {
			rr.Hdr.Ttl &^= _DO
		}
	} else {
		rr.Hdr.Ttl |= _DO
	}
}

// EDNS0 defines an EDNS0 Option. An OPT RR can have multiple options appended to it.
type EDNS0 interface {
	// Option returns the option code for the option.
	Option() uint16
	// pack returns the bytes of the option data.
	pack() ([]byte, error)
	// unpack sets the data as found in the buffer. Is also sets
	// the length of the slice as the length of the option data.
	unpack([]byte) error
	// String returns the string representation of the option.
	String() string
	// copy returns a deep-copy of the option.
	copy() EDNS0
}

// EDNS0_NSID option is used to retrieve a nameserver
// identifier. When sending a request Nsid must be set to the empty string
// The identifier is an opaque string encoded as hex.
// Basic use pattern for creating an nsid option:
//
//	o := new(dns.OPT)
//	o.Hdr.Name = "."
//	o.Hdr.Rrtype = dns.TypeOPT
//	e := new(dns.EDNS0_NSID)
//	e.Code = dns.EDNS0NSID
//	e.Nsid = "AA"
//	o.Option = append(o.Option, e)
type EDNS0_NSID struct {
	Code uint16 // Always EDNS0NSID
	Nsid string // This string needs to be hex encoded
}

func (e *EDNS0_NSID) pack() ([]byte, error) {
	h, err := hex.DecodeString(e.Nsid)
	if err != nil {
		return nil, err
	}
	return h, nil
}

// Option implements the EDNS0 interface.
func (e *EDNS0_NSID) Option() uint16        { return EDNS0NSID } // Option returns the option code.
func (e *EDNS0_NSID) unpack(b []byte) error { e.Nsid = hex.EncodeToString(b); return nil }
func (e *EDNS0_NSID) String() string        { return e.Nsid }
func (e *EDNS0_NSID) copy() EDNS0           { return &EDNS0_NSID{e.Code, e.Nsid} }

// EDNS0_SUBNET is the subnet option that is used to give the remote nameserver
// an idea of where the client lives. See RFC 7871. It can then give back a different
// answer depending on the location or network topology.
// Basic use pattern for creating an subnet option:
//
//	o := new(dns.OPT)
//	o.Hdr.Name = "."
//	o.Hdr.Rrtype = dns.TypeOPT
//	e := new(dns.EDNS0_SUBNET)
//	e.Code = dns.EDNS0SUBNET
//	e.Family = 1	// 1 for IPv4 source address, 2 for IPv6
//	e.SourceNetmask = 32	// 32 for IPV4, 128 for IPv6
//	e.SourceScope = 0
//	e.Address = net.ParseIP("127.0.0.1").To4()	// for IPv4
//	// e.Address = net.ParseIP("2001:7b8:32a::2")	// for IPV6
//	o.Option = append(o.Option, e)
//
// This code will parse all the available bits when unpacking (up to optlen).
// When packing it will apply SourceNetmask. If you need more advanced logic,
// patches welcome and good luck.
type EDNS0_SUBNET struct {
	Code          uint16 // Always EDNS0SUBNET
	Family        uint16 // 1 for IP, 2 for IP6
	SourceNetmask uint8
	SourceScope   uint8
	Address       net.IP
}

// Option implements the EDNS0 interface.
func (e *EDNS0_SUBNET) Option() uint16 { return EDNS0SUBNET }

func (e *EDNS0_SUBNET) pack() ([]byte, error) {
	b := make([]byte, 4)
	binary.BigEndian.PutUint16(b[0:], e.Family)
	b[2] = e.SourceNetmask
	b[3] = e.SourceScope
	switch e.Family {
	case 0:
		// "dig" sets AddressFamily to 0 if SourceNetmask is also 0
		// We might don't need to complain either
		if e.SourceNetmask != 0 {
			return nil, errors.New("dns: bad address family")
		}
	case 1:
		if e.SourceNetmask > net.IPv4len*8 {
			return nil, errors.New("dns: bad netmask")
		}
		if len(e.Address.To4()) != net.IPv4len {
			return nil, errors.New("dns: bad address")
		}
		ip := e.Address.To4().Mask(net.CIDRMask(int(e.SourceNetmask), net.IPv4len*8))
		needLength := (e.SourceNetmask + 8 - 1) / 8 // division rounding up
		b = append(b, ip[:needLength]...)
	case 2:
		if e.SourceNetmask > net.IPv6len*8 {
			return nil, errors.New("dns: bad netmask")
		}
		if len(e.Address) != net.IPv6len {
			return nil, errors.New("dns: bad address")
		}
		ip := e.Address.Mask(net.CIDRMask(int(e.SourceNetmask), net.IPv6len*8))
		needLength := (e.SourceNetmask + 8 - 1) / 8 // division rounding up
		b = append(b, ip[:needLength]...)
	default:
		return nil, errors.New("dns: bad address family")
	}
	return b, nil
}

func (e *EDNS0_SUBNET) unpack(b []byte) error {
	if len(b) < 4 {
		return ErrBuf
	}
	e.Family = binary.BigEndian.Uint16(b)
	e.SourceNetmask = b[2]
	e.SourceScope = b[3]
	switch e.Family {
	case 0:
		// "dig" sets AddressFamily to 0 if SourceNetmask is also 0
		// It's okay to accept such a packet
		if e.SourceNetmask != 0 {
			return errors.New("dns: bad address family")
		}
		e.Address = net.IPv4(0, 0, 0, 0)
	case 1:
		if e.SourceNetmask > net.IPv4len*8 || e.SourceScope > net.IPv4len*8 {
			return errors.New("dns: bad netmask")
		}
		addr := make(net.IP, net.IPv4len)
		copy(addr, b[4:])
		e.Address = addr.To16()
	case 2:
		if e.SourceNetmask > net.IPv6len*8 || e.SourceScope > net.IPv6len*8 {
			return errors.New("dns: bad netmask")
		}
		addr := make(net.IP, net.IPv6len)
		copy(addr, b[4:])
		e.Address = addr
	default:
		return errors.New("dns: bad address family")
	}
	return nil
}

func (e *EDNS0_SUBNET) String() (s string) {
	if e.Address == nil {
		s = "<nil>"
	} else if e.Address.To4() != nil {
		s = e.Address.String()
	} else {
		s = "[" + e.Address.String() + "]"
	}
	s += "/" + strconv.Itoa(int(e.SourceNetmask)) + "/" + strconv.Itoa(int(e.SourceScope))
	return
}

func (e *EDNS0_SUBNET) copy() EDNS0 {
	return &EDNS0_SUBNET{
		e.Code,
		e.Family,
		e.SourceNetmask,
		e.SourceScope,
		e.Address,
	}
}

// The EDNS0_COOKIE option is used to add a DNS Cookie to a message.
//
//	o := new(dns.OPT)
//	o.Hdr.Name = "."
//	o.Hdr.Rrtype = dns.TypeOPT
//	e := new(dns.EDNS0_COOKIE)
//	e.Code = dns.EDNS0COOKIE
//	e.Cookie = "24a5ac.."
//	o.Option = append(o.Option, e)
//
// The Cookie field consists out of a client cookie (RFC 7873 Section 4), that is
// always 8 bytes. It may then optionally be followed by the server cookie. The server
// cookie is of variable length, 8 to a maximum of 32 bytes. In other words:
//
//	cCookie := o.Cookie[:16]
//	sCookie := o.Cookie[16:]
//
// There is no guarantee that the Cookie string has a specific length.
type EDNS0_COOKIE struct {
	Code   uint16 // Always EDNS0COOKIE
	Cookie string // Hex-encoded cookie data
}

func (e *EDNS0_COOKIE) pack() ([]byte, error) {
	h, err := hex.DecodeString(e.Cookie)
	if err != nil {
		return nil, err
	}
	return h, nil
}

// Option implements the EDNS0 interface.
func (e *EDNS0_COOKIE) Option() uint16        { return EDNS0COOKIE }
func (e *EDNS0_COOKIE) unpack(b []byte) error { e.Cookie = hex.EncodeToString(b); return nil }
func (e *EDNS0_COOKIE) String() string        { return e.Cookie }
func (e *EDNS0_COOKIE) copy() EDNS0           { return &EDNS0_COOKIE{e.Code, e.Cookie} }

// The EDNS0_UL (Update Lease) (draft RFC) option is used to tell the server to set
// an expiration on an update RR. This is helpful for clients that cannot clean
// up after themselves. This is a draft RFC and more information can be found at
// https://tools.ietf.org/html/draft-sekar-dns-ul-02
//
//	o := new(dns.OPT)
//	o.Hdr.Name = "."
//	o.Hdr.Rrtype = dns.TypeOPT
//	e := new(dns.EDNS0_UL)
//	e.Code = dns.EDNS0UL
//	e.Lease = 120 // in seconds
//	o.Option = append(o.Option, e)
type EDNS0_UL struct {
	Code     uint16 // Always EDNS0UL
	Lease    uint32
	KeyLease uint32
}

// Option implements the EDNS0 interface.
func (e *EDNS0_UL) Option() uint16 { return EDNS0UL }
func (e *EDNS0_UL) String() string { return fmt.Sprintf("%d %d", e.Lease, e.KeyLease) }
func (e *EDNS0_UL) copy() EDNS0    { return &EDNS0_UL{e.Code, e.Lease, e.KeyLease} }

// Copied: http://golang.org/src/pkg/net/dnsmsg.go
func (e *EDNS0_UL) pack() ([]byte, error) {
	var b []byte
	if e.KeyLease == 0 {
		b = make([]byte, 4)
	} else {
		b = make([]byte, 8)
		binary.BigEndian.PutUint32(b[4:], e.KeyLease)
	}
	binary.BigEndian.PutUint32(b, e.Lease)
	return b, nil
}

func (e *EDNS0_UL) unpack(b []byte) error {
	switch len(b) {
	case 4:
		e.KeyLease = 0
	case 8:
		e.KeyLease = binary.BigEndian.Uint32(b[4:])
	default:
		return ErrBuf
	}
	e.Lease = binary.BigEndian.Uint32(b)
	return nil
}

// EDNS0_LLQ stands for Long Lived Queries: http://tools.ietf.org/html/draft-sekar-dns-llq-01
// Implemented for completeness, as the EDNS0 type code is assigned.
type EDNS0_LLQ struct {
	Code      uint16 // Always EDNS0LLQ
	Version   uint16
	Opcode    uint16
	Error     uint16
	Id        uint64
	LeaseLife uint32
}

// Option implements the EDNS0 interface.
func (e *EDNS0_LLQ) Option() uint16 { return EDNS0LLQ }

func (e *EDNS0_LLQ) pack() ([]byte, error) {
	b := make([]byte, 18)
	binary.BigEndian.PutUint16(b[0:], e.Version)
	binary.BigEndian.PutUint16(b[2:], e.Opcode)
	binary.BigEndian.PutUint16(b[4:], e.Error)
	binary.BigEndian.PutUint64(b[6:], e.Id)
	binary.BigEndian.PutUint32(b[14:], e.LeaseLife)
	return b, nil
}

func (e *EDNS0_LLQ) unpack(b []byte) error {
	if len(b) < 18 {
		return ErrBuf
	}
	e.Version = binary.BigEndian.Uint16(b[0:])
	e.Opcode = binary.BigEndian.Uint16(b[2:])
	e.Error = binary.BigEndian.Uint16(b[4:])
	e.Id = binary.BigEndian.Uint64(b[6:])
	e.LeaseLife = binary.BigEndian.Uint32(b[14:])
	return nil
}

func (e *EDNS0_LLQ) String() string {
	s := strconv.FormatUint(uint64(e.Version), 10) + " " + strconv.FormatUint(uint64(e.Opcode), 10) +
		" " + strconv.FormatUint(uint64(e.Error), 10) + " " + strconv.FormatUint(e.Id, 10) +
		" " + strconv.FormatUint(uint64(e.LeaseLife), 10)
	return s
}
func (e *EDNS0_LLQ) copy() EDNS0 {
	return &EDNS0_LLQ{e.Code, e.Version, e.Opcode, e.Error, e.Id, e.LeaseLife}
}

// EDNS0_DUA implements the EDNS0 "DNSSEC Algorithm Understood" option. See RFC 6975.
type EDNS0_DAU struct {
	Code    uint16 // Always EDNS0DAU
	AlgCode []uint8
}

// Option implements the EDNS0 interface.
func (e *EDNS0_DAU) Option() uint16        { return EDNS0DAU }
func (e *EDNS0_DAU) pack() ([]byte, error) { return e.AlgCode, nil }
func (e *EDNS0_DAU) unpack(b []byte) error { e.AlgCode = b; return nil }

func (e *EDNS0_DAU) String() string {
	s := ""
	for _, alg := range e.AlgCode {
		if a, ok := AlgorithmToString[alg]; ok {
			s += " " + a
		} else {
			s += " " + strconv.Itoa(int(alg))
		}
	}
	return s
}
func (e *EDNS0_DAU) copy() EDNS0 { return &EDNS0_DAU{e.Code, e.AlgCode} }

// EDNS0_DHU implements the EDNS0 "DS Hash Understood" option. See RFC 6975.
type EDNS0_DHU struct {
	Code    uint16 // Always EDNS0DHU
	AlgCode []uint8
}

// Option implements the EDNS0 interface.
func (e *EDNS0_DHU) Option() uint16        { return EDNS0DHU }
func (e *EDNS0_DHU) pack() ([]byte, error) { return e.AlgCode, nil }
func (e *EDNS0_DHU) unpack(b []byte) error { e.AlgCode = b; return nil }

func (e *EDNS0_DHU) String() string {
	s := ""
	for _, alg := range e.AlgCode {
		if a, ok := HashToString[alg]; ok {
			s += " " + a
		} else {
			s += " " + strconv.Itoa(int(alg))
		}
	}
	return s
}
func (e *EDNS0_DHU) copy() EDNS0 { return &EDNS0_DHU{e.Code, e.AlgCode} }

// EDNS0_N3U implements the EDNS0 "NSEC3 Hash Understood" option. See RFC 6975.
type EDNS0_N3U struct {
	Code    uint16 // Always EDNS0N3U
	AlgCode []uint8
}

// Option implements the EDNS0 interface.
func (e *EDNS0_N3U) Option() uint16        { return EDNS0N3U }
func (e *EDNS0_N3U) pack() ([]byte, error) { return e.AlgCode, nil }
func (e *EDNS0_N3U) unpack(b []byte) error { e.AlgCode = b; return nil }

func (e *EDNS0_N3U) String() string {
	// Re-use the hash map
	s := ""
	for _, alg := range e.AlgCode {
		if a, ok := HashToString[alg]; ok {
			s += " " + a
		} else {
			s += " " + strconv.Itoa(int(alg))
		}
	}
	return s
}
func (e *EDNS0_N3U) copy() EDNS0 { return &EDNS0_N3U{e.Code, e.AlgCode} }

// EDNS0_EXPIRE implementes the EDNS0 option as described in RFC 7314.
type EDNS0_EXPIRE struct {
	Code   uint16 // Always EDNS0EXPIRE
	Expire uint32
}

// Option implements the EDNS0 interface.
func (e *EDNS0_EXPIRE) Option() uint16 { return EDNS0EXPIRE }
func (e *EDNS0_EXPIRE) String() string { return strconv.FormatUint(uint64(e.Expire), 10) }
func (e *EDNS0_EXPIRE) copy() EDNS0    { return &EDNS0_EXPIRE{e.Code, e.Expire} }

func (e *EDNS0_EXPIRE) pack() ([]byte, error) {
	b := make([]byte, 4)
	binary.BigEndian.PutUint32(b, e.Expire)
	return b, nil
}

func (e *EDNS0_EXPIRE) unpack(b []byte) error {
	if len(b) == 0 {
		// zero-length EXPIRE query, see RFC 7314 Section 2
		return nil
	}
	if len(b) < 4 {
		return ErrBuf
	}
	e.Expire = binary.BigEndian.Uint32(b)
	return nil
}

// The EDNS0_LOCAL option is used for local/experimental purposes. The option
// code is recommended to be within the range [EDNS0LOCALSTART, EDNS0LOCALEND]
// (RFC6891), although any unassigned code can actually be used.  The content of
// the option is made available in Data, unaltered.
// Basic use pattern for creating a local option:
//
//	o := new(dns.OPT)
//	o.Hdr.Name = "."
//	o.Hdr.Rrtype = dns.TypeOPT
//	e := new(dns.EDNS0_LOCAL)
//	e.Code = dns.EDNS0LOCALSTART
//	e.Data = []byte{72, 82, 74}
//	o.Option = append(o.Option, e)
type EDNS0_LOCAL struct {
	Code uint16
	Data []byte
}

// Option implements the EDNS0 interface.
func (e *EDNS0_LOCAL) Option() uint16 { return e.Code }
func (e *EDNS0_LOCAL) String() string {
	return strconv.FormatInt(int64(e.Code), 10) + ":0x" + hex.EncodeToString(e.Data)
}
func (e *EDNS0_LOCAL) copy() EDNS0 {
	b := make([]byte, len(e.Data))
	copy(b, e.Data)
	return &EDNS0_LOCAL{e.Code, b}
}

func (e *EDNS0_LOCAL) pack() ([]byte, error) {
	b := make([]byte, len(e.Data))
	copied := copy(b, e.Data)
	if copied != len(e.Data) {
		return nil, ErrBuf
	}
	return b, nil
}

func (e *EDNS0_LOCAL) unpack(b []byte) error {
	e.Data = make([]byte, len(b))
	copied := copy(e.Data, b)
	if copied != len(b) {
		return ErrBuf
	}
	return nil
}

// EDNS0_TCP_KEEPALIVE is an EDNS0 option that instructs the server to keep
// the TCP connection alive. See RFC 7828.
type EDNS0_TCP_KEEPALIVE struct {
	Code    uint16 // Always EDNSTCPKEEPALIVE
	Length  uint16 // the value 0 if the TIMEOUT is omitted, the value 2 if it is present;
	Timeout uint16 // an idle timeout value for the TCP connection, specified in units of 100 milliseconds, encoded in network byte order.
}

// Option implements the EDNS0 interface.
func (e *EDNS0_TCP_KEEPALIVE) Option() uint16 { return EDNS0TCPKEEPALIVE }

func (e *EDNS0_TCP_KEEPALIVE) pack() ([]byte, error) {
	if e.Timeout != 0 && e.Length != 2 {
		return nil, errors.New("dns: timeout specified but length is not 2")
	}
	if e.Timeout == 0 && e.Length != 0 {
		return nil, errors.New("dns: timeout not specified but length is not 0")
	}
	b := make([]byte, 4+e.Length)
	binary.BigEndian.PutUint16(b[0:], e.Code)
	binary.BigEndian.PutUint16(b[2:], e.Length)
	if e.Length == 2 {
		binary.BigEndian.PutUint16(b[4:], e.Timeout)
	}
	return b, nil
}

func (e *EDNS0_TCP_KEEPALIVE) unpack(b []byte) error {
	if len(b) < 4 {
		return ErrBuf
	}
	e.Length = binary.BigEndian.Uint16(b[2:4])
	if e.Length != 0 && e.Length != 2 {
		return errors.New("dns: length mismatch, want 0/2 but got " + strconv.FormatUint(uint64(e.Length), 10))
	}
	if e.Length == 2 {
		if len(b) < 6 {
			return ErrBuf
		}
		e.Timeout = binary.BigEndian.Uint16(b[4:6])
	}
	return nil
}

func (e *EDNS0_TCP_KEEPALIVE) String() (s string) {
	s = "use tcp keep-alive"
	if e.Length == 0 {
		s += ", timeout omitted"
	} else {
		s += fmt.Sprintf(", timeout %dms", e.Timeout*100)
	}
	return
}
func (e *EDNS0_TCP_KEEPALIVE) copy() EDNS0 { return &EDNS0_TCP_KEEPALIVE{e.Code, e.Length, e.Timeout} }

// EDNS0_PADDING option is used to add padding to a request/response. The default
// value of padding SHOULD be 0x0 but other values MAY be used, for instance if
// compression is applied before encryption which may break signatures.
type EDNS0_PADDING struct {
	Padding []byte
}

// Option implements the EDNS0 interface.
func (e *EDNS0_PADDING) Option() uint16        { return EDNS0PADDING }
func (e *EDNS0_PADDING) pack() ([]byte, error) { return e.Padding, nil }
func (e *EDNS0_PADDING) unpack(b []byte) error { e.Padding = b; return nil }
func (e *EDNS0_PADDING) String() string        { return fmt.Sprintf("%0X", e.Padding) }
func (e *EDNS0_PADDING) copy() EDNS0 {
	b := make([]byte, len(e.Padding))
	copy(b, e.Padding)
	return &EDNS0_PADDING{b}
}
