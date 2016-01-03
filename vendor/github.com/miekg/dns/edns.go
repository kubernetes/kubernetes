// EDNS0
//
// EDNS0 is an extension mechanism for the DNS defined in RFC 2671 and updated
// by RFC 6891. It defines an new RR type, the OPT RR, which is then completely
// abused.
// Basic use pattern for creating an (empty) OPT RR:
//
//	o := new(dns.OPT)
//	o.Hdr.Name = "." // MUST be the root zone, per definition.
//	o.Hdr.Rrtype = dns.TypeOPT
//
// The rdata of an OPT RR consists out of a slice of EDNS0 (RFC 6891)
// interfaces. Currently only a few have been standardized: EDNS0_NSID
// (RFC 5001) and EDNS0_SUBNET (draft-vandergaast-edns-client-subnet-02). Note
// that these options may be combined in an OPT RR.
// Basic use pattern for a server to check if (and which) options are set:
//
//	// o is a dns.OPT
//	for _, s := range o.Option {
//		switch e := s.(type) {
//		case *dns.EDNS0_NSID:
//			// do stuff with e.Nsid
//		case *dns.EDNS0_SUBNET:
//			// access e.Family, e.Address, etc.
//		}
//	}
package dns

import (
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
	EDNS0SUBNETDRAFT = 0x50fa  // Don't use! Use EDNS0SUBNET
	_DO              = 1 << 15 // dnssec ok
)

type OPT struct {
	Hdr    RR_Header
	Option []EDNS0 `dns:"opt"`
}

func (rr *OPT) Header() *RR_Header {
	return &rr.Hdr
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
		}
	}
	return s
}

func (rr *OPT) len() int {
	l := rr.Hdr.len()
	for i := 0; i < len(rr.Option); i++ {
		lo, _ := rr.Option[i].pack()
		l += 2 + len(lo)
	}
	return l
}

func (rr *OPT) copy() RR {
	return &OPT{*rr.Hdr.copyHeader(), rr.Option}
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
func (rr *OPT) ExtendedRcode() uint8 {
	return uint8((rr.Hdr.Ttl & 0xFF000000) >> 24)
}

// SetExtendedRcode sets the EDNS extended RCODE field.
func (rr *OPT) SetExtendedRcode(v uint8) {
	rr.Hdr.Ttl = rr.Hdr.Ttl&0x00FFFFFF | (uint32(v) << 24)
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

// EDNS0 defines an EDNS0 Option. An OPT RR can have multiple options appended to
// it.
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

// The subnet EDNS0 option is used to give the remote nameserver
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
	b[0], b[1] = packUint16(e.Family)
	b[2] = e.SourceNetmask
	b[3] = e.SourceScope
	switch e.Family {
	case 1:
		if e.SourceNetmask > net.IPv4len*8 {
			return nil, errors.New("dns: bad netmask")
		}
		ip := make([]byte, net.IPv4len)
		a := e.Address.To4().Mask(net.CIDRMask(int(e.SourceNetmask), net.IPv4len*8))
		for i := 0; i < net.IPv4len; i++ {
			if i+1 > len(e.Address) {
				break
			}
			ip[i] = a[i]
		}
		needLength := e.SourceNetmask / 8
		if e.SourceNetmask%8 > 0 {
			needLength++
		}
		ip = ip[:needLength]
		b = append(b, ip...)
	case 2:
		if e.SourceNetmask > net.IPv6len*8 {
			return nil, errors.New("dns: bad netmask")
		}
		ip := make([]byte, net.IPv6len)
		a := e.Address.Mask(net.CIDRMask(int(e.SourceNetmask), net.IPv6len*8))
		for i := 0; i < net.IPv6len; i++ {
			if i+1 > len(e.Address) {
				break
			}
			ip[i] = a[i]
		}
		needLength := e.SourceNetmask / 8
		if e.SourceNetmask%8 > 0 {
			needLength++
		}
		ip = ip[:needLength]
		b = append(b, ip...)
	default:
		return nil, errors.New("dns: bad address family")
	}
	return b, nil
}

func (e *EDNS0_SUBNET) unpack(b []byte) error {
	lb := len(b)
	if lb < 4 {
		return ErrBuf
	}
	e.Family, _ = unpackUint16(b, 0)
	e.SourceNetmask = b[2]
	e.SourceScope = b[3]
	switch e.Family {
	case 1:
		addr := make([]byte, 4)
		for i := 0; i < int(e.SourceNetmask/8); i++ {
			if i >= len(addr) || 4+i >= len(b) {
				return ErrBuf
			}
			addr[i] = b[4+i]
		}
		e.Address = net.IPv4(addr[0], addr[1], addr[2], addr[3])
	case 2:
		addr := make([]byte, 16)
		for i := 0; i < int(e.SourceNetmask/8); i++ {
			if i >= len(addr) || 4+i >= len(b) {
				return ErrBuf
			}
			addr[i] = b[4+i]
		}
		e.Address = net.IP{addr[0], addr[1], addr[2], addr[3], addr[4],
			addr[5], addr[6], addr[7], addr[8], addr[9], addr[10],
			addr[11], addr[12], addr[13], addr[14], addr[15]}
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

// The UL (Update Lease) EDNS0 (draft RFC) option is used to tell the server to set
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
	b[0] = byte(e.Lease >> 24)
	b[1] = byte(e.Lease >> 16)
	b[2] = byte(e.Lease >> 8)
	b[3] = byte(e.Lease)
	return b, nil
}

func (e *EDNS0_UL) unpack(b []byte) error {
	if len(b) < 4 {
		return ErrBuf
	}
	e.Lease = uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
	return nil
}

// Long Lived Queries: http://tools.ietf.org/html/draft-sekar-dns-llq-01
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
	b[0], b[1] = packUint16(e.Version)
	b[2], b[3] = packUint16(e.Opcode)
	b[4], b[5] = packUint16(e.Error)
	b[6] = byte(e.Id >> 56)
	b[7] = byte(e.Id >> 48)
	b[8] = byte(e.Id >> 40)
	b[9] = byte(e.Id >> 32)
	b[10] = byte(e.Id >> 24)
	b[11] = byte(e.Id >> 16)
	b[12] = byte(e.Id >> 8)
	b[13] = byte(e.Id)
	b[14] = byte(e.LeaseLife >> 24)
	b[15] = byte(e.LeaseLife >> 16)
	b[16] = byte(e.LeaseLife >> 8)
	b[17] = byte(e.LeaseLife)
	return b, nil
}

func (e *EDNS0_LLQ) unpack(b []byte) error {
	if len(b) < 18 {
		return ErrBuf
	}
	e.Version, _ = unpackUint16(b, 0)
	e.Opcode, _ = unpackUint16(b, 2)
	e.Error, _ = unpackUint16(b, 4)
	e.Id = uint64(b[6])<<56 | uint64(b[6+1])<<48 | uint64(b[6+2])<<40 |
		uint64(b[6+3])<<32 | uint64(b[6+4])<<24 | uint64(b[6+5])<<16 | uint64(b[6+6])<<8 | uint64(b[6+7])
	e.LeaseLife = uint32(b[14])<<24 | uint32(b[14+1])<<16 | uint32(b[14+2])<<8 | uint32(b[14+3])
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
	e.Expire = uint32(b[0])<<24 | uint32(b[1])<<16 | uint32(b[2])<<8 | uint32(b[3])
	return nil
}
