package dns

import (
	"encoding/binary"
	"encoding/hex"
	"errors"
	"net"
	"strconv"
)

// EDNS0 Option codes.
const (
	EDNS0LLQ         = 0x1     // long lived queries: http://tools.ietf.org/html/draft-sekar-dns-llq-01
	EDNS0UL          = 0x2     // update lease draft: http://files.dns-sd.org/draft-sekar-dns-ul.txt
	EDNS0NSID        = 0x3     // nsid (RFC5001)
	EDNS0DAU         = 0x5     // DNSSEC Algorithm Understood
	EDNS0DHU         = 0x6     // DS Hash Understood
	EDNS0N3U         = 0x7     // NSEC3 Hash Understood
	EDNS0SUBNET      = 0x8     // client-subnet (RFC6891)
	EDNS0EXPIRE      = 0x9     // EDNS0 expire
	EDNS0COOKIE      = 0xa     // EDNS0 Cookie
	EDNS0SUBNETDRAFT = 0x50fa  // Don't use! Use EDNS0SUBNET
	EDNS0LOCALSTART  = 0xFDE9  // Beginning of range reserved for local/experimental use (RFC6891)
	EDNS0LOCALEND    = 0xFFFE  // End of range reserved for local/experimental use (RFC6891)
	_DO              = 1 << 15 // dnssec ok
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
			if o.(*EDNS0_SUBNET).DraftOption {
				s += " (draft)"
			}
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
		}
	}
	return s
}

func (rr *OPT) len() int {
	l := rr.Hdr.len()
	for i := 0; i < len(rr.Option); i++ {
		l += 4 // Account for 2-byte option code and 2-byte option length.
		lo, _ := rr.Option[i].pack()
		l += len(lo)
	}
	return l
}

// return the old value -> delete SetVersion?

// Version returns the EDNS version used. Only zero is defined.
func (rr *OPT) Version() uint8 {
	return uint8((rr.Hdr.Ttl & 0x00FF0000) >> 16)
}

// SetVersion sets the version of EDNS. This is usually zero.
func (rr *OPT) SetVersion(v uint8) {
	rr.Hdr.Ttl = rr.Hdr.Ttl&0xFF00FFFF | (uint32(v) << 16)
}

// ExtendedRcode returns the EDNS extended RCODE field (the upper 8 bits of the TTL).
func (rr *OPT) ExtendedRcode() int {
	return int((rr.Hdr.Ttl&0xFF000000)>>24) + 15
}

// SetExtendedRcode sets the EDNS extended RCODE field.
func (rr *OPT) SetExtendedRcode(v uint8) {
	if v < RcodeBadVers { // Smaller than 16.. Use the 4 bits you have!
		return
	}
	rr.Hdr.Ttl = rr.Hdr.Ttl&0x00FFFFFF | (uint32(v-15) << 24)
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
func (rr *OPT) SetDo() {
	rr.Hdr.Ttl |= _DO
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
}

// The nsid EDNS0 option is used to retrieve a nameserver
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

func (e *EDNS0_NSID) Option() uint16        { return EDNS0NSID }
func (e *EDNS0_NSID) unpack(b []byte) error { e.Nsid = hex.EncodeToString(b); return nil }
func (e *EDNS0_NSID) String() string        { return string(e.Nsid) }

// EDNS0_SUBNET is the subnet option that is used to give the remote nameserver
// an idea of where the client lives. It can then give back a different
// answer depending on the location or network topology.
// Basic use pattern for creating an subnet option:
//
//	o := new(dns.OPT)
//	o.Hdr.Name = "."
//	o.Hdr.Rrtype = dns.TypeOPT
//	e := new(dns.EDNS0_SUBNET)
//	e.Code = dns.EDNS0SUBNET
//	e.Family = 1	// 1 for IPv4 source address, 2 for IPv6
//	e.NetMask = 32	// 32 for IPV4, 128 for IPv6
//	e.SourceScope = 0
//	e.Address = net.ParseIP("127.0.0.1").To4()	// for IPv4
//	// e.Address = net.ParseIP("2001:7b8:32a::2")	// for IPV6
//	o.Option = append(o.Option, e)
//
// Note: the spec (draft-ietf-dnsop-edns-client-subnet-00) has some insane logic
// for which netmask applies to the address. This code will parse all the
// available bits when unpacking (up to optlen). When packing it will apply
// SourceNetmask. If you need more advanced logic, patches welcome and good luck.
type EDNS0_SUBNET struct {
	Code          uint16 // Always EDNS0SUBNET
	Family        uint16 // 1 for IP, 2 for IP6
	SourceNetmask uint8
	SourceScope   uint8
	Address       net.IP
	DraftOption   bool // Set to true if using the old (0x50fa) option code
}

func (e *EDNS0_SUBNET) Option() uint16 {
	if e.DraftOption {
		return EDNS0SUBNETDRAFT
	}
	return EDNS0SUBNET
}

func (e *EDNS0_SUBNET) pack() ([]byte, error) {
	b := make([]byte, 4)
	binary.BigEndian.PutUint16(b[0:], e.Family)
	b[2] = e.SourceNetmask
	b[3] = e.SourceScope
	switch e.Family {
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
	case 1:
		if e.SourceNetmask > net.IPv4len*8 || e.SourceScope > net.IPv4len*8 {
			return errors.New("dns: bad netmask")
		}
		addr := make([]byte, net.IPv4len)
		for i := 0; i < net.IPv4len && 4+i < len(b); i++ {
			addr[i] = b[4+i]
		}
		e.Address = net.IPv4(addr[0], addr[1], addr[2], addr[3])
	case 2:
		if e.SourceNetmask > net.IPv6len*8 || e.SourceScope > net.IPv6len*8 {
			return errors.New("dns: bad netmask")
		}
		addr := make([]byte, net.IPv6len)
		for i := 0; i < net.IPv6len && 4+i < len(b); i++ {
			addr[i] = b[4+i]
		}
		e.Address = net.IP{addr[0], addr[1], addr[2], addr[3], addr[4],
			addr[5], addr[6], addr[7], addr[8], addr[9], addr[10],
			addr[11], addr[12], addr[13], addr[14], addr[15]}
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

// The Cookie EDNS0 option
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

func (e *EDNS0_COOKIE) Option() uint16        { return EDNS0COOKIE }
func (e *EDNS0_COOKIE) unpack(b []byte) error { e.Cookie = hex.EncodeToString(b); return nil }
func (e *EDNS0_COOKIE) String() string        { return e.Cookie }

// The EDNS0_UL (Update Lease) (draft RFC) option is used to tell the server to set
// an expiration on an update RR. This is helpful for clients that cannot clean
// up after themselves. This is a draft RFC and more information can be found at
// http://files.dns-sd.org/draft-sekar-dns-ul.txt
//
//	o := new(dns.OPT)
//	o.Hdr.Name = "."
//	o.Hdr.Rrtype = dns.TypeOPT
//	e := new(dns.EDNS0_UL)
//	e.Code = dns.EDNS0UL
//	e.Lease = 120 // in seconds
//	o.Option = append(o.Option, e)
type EDNS0_UL struct {
	Code  uint16 // Always EDNS0UL
	Lease uint32
}

func (e *EDNS0_UL) Option() uint16 { return EDNS0UL }
func (e *EDNS0_UL) String() string { return strconv.FormatUint(uint64(e.Lease), 10) }

// Copied: http://golang.org/src/pkg/net/dnsmsg.go
func (e *EDNS0_UL) pack() ([]byte, error) {
	b := make([]byte, 4)
	binary.BigEndian.PutUint32(b, e.Lease)
	return b, nil
}

func (e *EDNS0_UL) unpack(b []byte) error {
	if len(b) < 4 {
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
		" " + strconv.FormatUint(uint64(e.Error), 10) + " " + strconv.FormatUint(uint64(e.Id), 10) +
		" " + strconv.FormatUint(uint64(e.LeaseLife), 10)
	return s
}

type EDNS0_DAU struct {
	Code    uint16 // Always EDNS0DAU
	AlgCode []uint8
}

func (e *EDNS0_DAU) Option() uint16        { return EDNS0DAU }
func (e *EDNS0_DAU) pack() ([]byte, error) { return e.AlgCode, nil }
func (e *EDNS0_DAU) unpack(b []byte) error { e.AlgCode = b; return nil }

func (e *EDNS0_DAU) String() string {
	s := ""
	for i := 0; i < len(e.AlgCode); i++ {
		if a, ok := AlgorithmToString[e.AlgCode[i]]; ok {
			s += " " + a
		} else {
			s += " " + strconv.Itoa(int(e.AlgCode[i]))
		}
	}
	return s
}

type EDNS0_DHU struct {
	Code    uint16 // Always EDNS0DHU
	AlgCode []uint8
}

func (e *EDNS0_DHU) Option() uint16        { return EDNS0DHU }
func (e *EDNS0_DHU) pack() ([]byte, error) { return e.AlgCode, nil }
func (e *EDNS0_DHU) unpack(b []byte) error { e.AlgCode = b; return nil }

func (e *EDNS0_DHU) String() string {
	s := ""
	for i := 0; i < len(e.AlgCode); i++ {
		if a, ok := HashToString[e.AlgCode[i]]; ok {
			s += " " + a
		} else {
			s += " " + strconv.Itoa(int(e.AlgCode[i]))
		}
	}
	return s
}

type EDNS0_N3U struct {
	Code    uint16 // Always EDNS0N3U
	AlgCode []uint8
}

func (e *EDNS0_N3U) Option() uint16        { return EDNS0N3U }
func (e *EDNS0_N3U) pack() ([]byte, error) { return e.AlgCode, nil }
func (e *EDNS0_N3U) unpack(b []byte) error { e.AlgCode = b; return nil }

func (e *EDNS0_N3U) String() string {
	// Re-use the hash map
	s := ""
	for i := 0; i < len(e.AlgCode); i++ {
		if a, ok := HashToString[e.AlgCode[i]]; ok {
			s += " " + a
		} else {
			s += " " + strconv.Itoa(int(e.AlgCode[i]))
		}
	}
	return s
}

type EDNS0_EXPIRE struct {
	Code   uint16 // Always EDNS0EXPIRE
	Expire uint32
}

func (e *EDNS0_EXPIRE) Option() uint16 { return EDNS0EXPIRE }
func (e *EDNS0_EXPIRE) String() string { return strconv.FormatUint(uint64(e.Expire), 10) }

func (e *EDNS0_EXPIRE) pack() ([]byte, error) {
	b := make([]byte, 4)
	b[0] = byte(e.Expire >> 24)
	b[1] = byte(e.Expire >> 16)
	b[2] = byte(e.Expire >> 8)
	b[3] = byte(e.Expire)
	return b, nil
}

func (e *EDNS0_EXPIRE) unpack(b []byte) error {
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

func (e *EDNS0_LOCAL) Option() uint16 { return e.Code }
func (e *EDNS0_LOCAL) String() string {
	return strconv.FormatInt(int64(e.Code), 10) + ":0x" + hex.EncodeToString(e.Data)
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
