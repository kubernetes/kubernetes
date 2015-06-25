package dns

import (
	"encoding/base64"
	"fmt"
	"net"
	"strconv"
	"strings"
	"time"
)

type (
	// Type is a DNS type.
	Type uint16
	// Class is a DNS class.
	Class uint16
	// Name is a DNS domain name.
	Name string
)

// Packet formats

// Wire constants and supported types.
const (
	// valid RR_Header.Rrtype and Question.qtype

	TypeNone       uint16 = 0
	TypeA          uint16 = 1
	TypeNS         uint16 = 2
	TypeMD         uint16 = 3
	TypeMF         uint16 = 4
	TypeCNAME      uint16 = 5
	TypeSOA        uint16 = 6
	TypeMB         uint16 = 7
	TypeMG         uint16 = 8
	TypeMR         uint16 = 9
	TypeNULL       uint16 = 10
	TypeWKS        uint16 = 11
	TypePTR        uint16 = 12
	TypeHINFO      uint16 = 13
	TypeMINFO      uint16 = 14
	TypeMX         uint16 = 15
	TypeTXT        uint16 = 16
	TypeRP         uint16 = 17
	TypeAFSDB      uint16 = 18
	TypeX25        uint16 = 19
	TypeISDN       uint16 = 20
	TypeRT         uint16 = 21
	TypeNSAP       uint16 = 22
	TypeNSAPPTR    uint16 = 23
	TypeSIG        uint16 = 24
	TypeKEY        uint16 = 25
	TypePX         uint16 = 26
	TypeGPOS       uint16 = 27
	TypeAAAA       uint16 = 28
	TypeLOC        uint16 = 29
	TypeNXT        uint16 = 30
	TypeEID        uint16 = 31
	TypeNIMLOC     uint16 = 32
	TypeSRV        uint16 = 33
	TypeATMA       uint16 = 34
	TypeNAPTR      uint16 = 35
	TypeKX         uint16 = 36
	TypeCERT       uint16 = 37
	TypeDNAME      uint16 = 39
	TypeOPT        uint16 = 41 // EDNS
	TypeDS         uint16 = 43
	TypeSSHFP      uint16 = 44
	TypeIPSECKEY   uint16 = 45
	TypeRRSIG      uint16 = 46
	TypeNSEC       uint16 = 47
	TypeDNSKEY     uint16 = 48
	TypeDHCID      uint16 = 49
	TypeNSEC3      uint16 = 50
	TypeNSEC3PARAM uint16 = 51
	TypeTLSA       uint16 = 52
	TypeHIP        uint16 = 55
	TypeNINFO      uint16 = 56
	TypeRKEY       uint16 = 57
	TypeTALINK     uint16 = 58
	TypeCDS        uint16 = 59
	TypeCDNSKEY    uint16 = 60
	TypeOPENPGPKEY uint16 = 61
	TypeSPF        uint16 = 99
	TypeUINFO      uint16 = 100
	TypeUID        uint16 = 101
	TypeGID        uint16 = 102
	TypeUNSPEC     uint16 = 103
	TypeNID        uint16 = 104
	TypeL32        uint16 = 105
	TypeL64        uint16 = 106
	TypeLP         uint16 = 107
	TypeEUI48      uint16 = 108
	TypeEUI64      uint16 = 109

	TypeTKEY uint16 = 249
	TypeTSIG uint16 = 250

	// valid Question.Qtype only

	TypeIXFR  uint16 = 251
	TypeAXFR  uint16 = 252
	TypeMAILB uint16 = 253
	TypeMAILA uint16 = 254
	TypeANY   uint16 = 255

	TypeURI      uint16 = 256
	TypeCAA      uint16 = 257
	TypeTA       uint16 = 32768
	TypeDLV      uint16 = 32769
	TypeReserved uint16 = 65535

	// valid Question.Qclass

	ClassINET   = 1
	ClassCSNET  = 2
	ClassCHAOS  = 3
	ClassHESIOD = 4
	ClassNONE   = 254
	ClassANY    = 255

	// Msg.rcode

	RcodeSuccess        = 0
	RcodeFormatError    = 1
	RcodeServerFailure  = 2
	RcodeNameError      = 3
	RcodeNotImplemented = 4
	RcodeRefused        = 5
	RcodeYXDomain       = 6
	RcodeYXRrset        = 7
	RcodeNXRrset        = 8
	RcodeNotAuth        = 9
	RcodeNotZone        = 10
	RcodeBadSig         = 16 // TSIG
	RcodeBadVers        = 16 // EDNS0
	RcodeBadKey         = 17
	RcodeBadTime        = 18
	RcodeBadMode        = 19 // TKEY
	RcodeBadName        = 20
	RcodeBadAlg         = 21
	RcodeBadTrunc       = 22 // TSIG

	// Opcode, there is no 3

	OpcodeQuery  = 0
	OpcodeIQuery = 1
	OpcodeStatus = 2
	OpcodeNotify = 4
	OpcodeUpdate = 5
)

// The wire format for the DNS packet header.
type Header struct {
	Id                                 uint16
	Bits                               uint16
	Qdcount, Ancount, Nscount, Arcount uint16
}

const (
	// Header.Bits
	_QR = 1 << 15 // query/response (response=1)
	_AA = 1 << 10 // authoritative
	_TC = 1 << 9  // truncated
	_RD = 1 << 8  // recursion desired
	_RA = 1 << 7  // recursion available
	_Z  = 1 << 6  // Z
	_AD = 1 << 5  // authticated data
	_CD = 1 << 4  // checking disabled

	LOC_EQUATOR       = 1 << 31 // RFC 1876, Section 2.
	LOC_PRIMEMERIDIAN = 1 << 31 // RFC 1876, Section 2.

	LOC_HOURS   = 60 * 1000
	LOC_DEGREES = 60 * LOC_HOURS

	LOC_ALTITUDEBASE = 100000
)

// RFC 4398, Section 2.1
const (
	CertPKIX = 1 + iota
	CertSPKI
	CertPGP
	CertIPIX
	CertISPKI
	CertIPGP
	CertACPKIX
	CertIACPKIX
	CertURI = 253
	CertOID = 254
)

var CertTypeToString = map[uint16]string{
	CertPKIX:    "PKIX",
	CertSPKI:    "SPKI",
	CertPGP:     "PGP",
	CertIPIX:    "IPIX",
	CertISPKI:   "ISPKI",
	CertIPGP:    "IPGP",
	CertACPKIX:  "ACPKIX",
	CertIACPKIX: "IACPKIX",
	CertURI:     "URI",
	CertOID:     "OID",
}

var StringToCertType = reverseInt16(CertTypeToString)

// Question holds a DNS question. There can be multiple questions in the
// question section of a message. Usually there is just one.
type Question struct {
	Name   string `dns:"cdomain-name"` // "cdomain-name" specifies encoding (and may be compressed)
	Qtype  uint16
	Qclass uint16
}

func (q *Question) String() (s string) {
	// prefix with ; (as in dig)
	s = ";" + sprintName(q.Name) + "\t"
	s += Class(q.Qclass).String() + "\t"
	s += " " + Type(q.Qtype).String()
	return s
}

func (q *Question) len() int {
	l := len(q.Name) + 1
	return l + 4
}

type ANY struct {
	Hdr RR_Header
	// Does not have any rdata
}

func (rr *ANY) Header() *RR_Header { return &rr.Hdr }
func (rr *ANY) copy() RR           { return &ANY{*rr.Hdr.copyHeader()} }
func (rr *ANY) String() string     { return rr.Hdr.String() }
func (rr *ANY) len() int           { return rr.Hdr.len() }

type CNAME struct {
	Hdr    RR_Header
	Target string `dns:"cdomain-name"`
}

func (rr *CNAME) Header() *RR_Header { return &rr.Hdr }
func (rr *CNAME) copy() RR           { return &CNAME{*rr.Hdr.copyHeader(), sprintName(rr.Target)} }
func (rr *CNAME) String() string     { return rr.Hdr.String() + rr.Target }
func (rr *CNAME) len() int           { return rr.Hdr.len() + len(rr.Target) + 1 }

type HINFO struct {
	Hdr RR_Header
	Cpu string
	Os  string
}

func (rr *HINFO) Header() *RR_Header { return &rr.Hdr }
func (rr *HINFO) copy() RR           { return &HINFO{*rr.Hdr.copyHeader(), rr.Cpu, rr.Os} }
func (rr *HINFO) String() string {
	return rr.Hdr.String() + sprintTxt([]string{rr.Cpu, rr.Os})
}
func (rr *HINFO) len() int { return rr.Hdr.len() + len(rr.Cpu) + len(rr.Os) }

type MB struct {
	Hdr RR_Header
	Mb  string `dns:"cdomain-name"`
}

func (rr *MB) Header() *RR_Header { return &rr.Hdr }
func (rr *MB) copy() RR           { return &MB{*rr.Hdr.copyHeader(), sprintName(rr.Mb)} }

func (rr *MB) String() string { return rr.Hdr.String() + rr.Mb }
func (rr *MB) len() int       { return rr.Hdr.len() + len(rr.Mb) + 1 }

type MG struct {
	Hdr RR_Header
	Mg  string `dns:"cdomain-name"`
}

func (rr *MG) Header() *RR_Header { return &rr.Hdr }
func (rr *MG) copy() RR           { return &MG{*rr.Hdr.copyHeader(), rr.Mg} }
func (rr *MG) len() int           { l := len(rr.Mg) + 1; return rr.Hdr.len() + l }
func (rr *MG) String() string     { return rr.Hdr.String() + sprintName(rr.Mg) }

type MINFO struct {
	Hdr   RR_Header
	Rmail string `dns:"cdomain-name"`
	Email string `dns:"cdomain-name"`
}

func (rr *MINFO) Header() *RR_Header { return &rr.Hdr }
func (rr *MINFO) copy() RR           { return &MINFO{*rr.Hdr.copyHeader(), rr.Rmail, rr.Email} }

func (rr *MINFO) String() string {
	return rr.Hdr.String() + sprintName(rr.Rmail) + " " + sprintName(rr.Email)
}

func (rr *MINFO) len() int {
	l := len(rr.Rmail) + 1
	n := len(rr.Email) + 1
	return rr.Hdr.len() + l + n
}

type MR struct {
	Hdr RR_Header
	Mr  string `dns:"cdomain-name"`
}

func (rr *MR) Header() *RR_Header { return &rr.Hdr }
func (rr *MR) copy() RR           { return &MR{*rr.Hdr.copyHeader(), rr.Mr} }
func (rr *MR) len() int           { l := len(rr.Mr) + 1; return rr.Hdr.len() + l }

func (rr *MR) String() string {
	return rr.Hdr.String() + sprintName(rr.Mr)
}

type MF struct {
	Hdr RR_Header
	Mf  string `dns:"cdomain-name"`
}

func (rr *MF) Header() *RR_Header { return &rr.Hdr }
func (rr *MF) copy() RR           { return &MF{*rr.Hdr.copyHeader(), rr.Mf} }
func (rr *MF) len() int           { return rr.Hdr.len() + len(rr.Mf) + 1 }

func (rr *MF) String() string {
	return rr.Hdr.String() + sprintName(rr.Mf)
}

type MD struct {
	Hdr RR_Header
	Md  string `dns:"cdomain-name"`
}

func (rr *MD) Header() *RR_Header { return &rr.Hdr }
func (rr *MD) copy() RR           { return &MD{*rr.Hdr.copyHeader(), rr.Md} }
func (rr *MD) len() int           { return rr.Hdr.len() + len(rr.Md) + 1 }

func (rr *MD) String() string {
	return rr.Hdr.String() + sprintName(rr.Md)
}

type MX struct {
	Hdr        RR_Header
	Preference uint16
	Mx         string `dns:"cdomain-name"`
}

func (rr *MX) Header() *RR_Header { return &rr.Hdr }
func (rr *MX) copy() RR           { return &MX{*rr.Hdr.copyHeader(), rr.Preference, rr.Mx} }
func (rr *MX) len() int           { l := len(rr.Mx) + 1; return rr.Hdr.len() + l + 2 }

func (rr *MX) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Mx)
}

type AFSDB struct {
	Hdr      RR_Header
	Subtype  uint16
	Hostname string `dns:"cdomain-name"`
}

func (rr *AFSDB) Header() *RR_Header { return &rr.Hdr }
func (rr *AFSDB) copy() RR           { return &AFSDB{*rr.Hdr.copyHeader(), rr.Subtype, rr.Hostname} }
func (rr *AFSDB) len() int           { l := len(rr.Hostname) + 1; return rr.Hdr.len() + l + 2 }

func (rr *AFSDB) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Subtype)) + " " + sprintName(rr.Hostname)
}

type X25 struct {
	Hdr         RR_Header
	PSDNAddress string
}

func (rr *X25) Header() *RR_Header { return &rr.Hdr }
func (rr *X25) copy() RR           { return &X25{*rr.Hdr.copyHeader(), rr.PSDNAddress} }
func (rr *X25) len() int           { return rr.Hdr.len() + len(rr.PSDNAddress) + 1 }

func (rr *X25) String() string {
	return rr.Hdr.String() + rr.PSDNAddress
}

type RT struct {
	Hdr        RR_Header
	Preference uint16
	Host       string `dns:"cdomain-name"`
}

func (rr *RT) Header() *RR_Header { return &rr.Hdr }
func (rr *RT) copy() RR           { return &RT{*rr.Hdr.copyHeader(), rr.Preference, rr.Host} }
func (rr *RT) len() int           { l := len(rr.Host) + 1; return rr.Hdr.len() + l + 2 }

func (rr *RT) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Host)
}

type NS struct {
	Hdr RR_Header
	Ns  string `dns:"cdomain-name"`
}

func (rr *NS) Header() *RR_Header { return &rr.Hdr }
func (rr *NS) len() int           { l := len(rr.Ns) + 1; return rr.Hdr.len() + l }
func (rr *NS) copy() RR           { return &NS{*rr.Hdr.copyHeader(), rr.Ns} }

func (rr *NS) String() string {
	return rr.Hdr.String() + sprintName(rr.Ns)
}

type PTR struct {
	Hdr RR_Header
	Ptr string `dns:"cdomain-name"`
}

func (rr *PTR) Header() *RR_Header { return &rr.Hdr }
func (rr *PTR) copy() RR           { return &PTR{*rr.Hdr.copyHeader(), rr.Ptr} }
func (rr *PTR) len() int           { l := len(rr.Ptr) + 1; return rr.Hdr.len() + l }

func (rr *PTR) String() string {
	return rr.Hdr.String() + sprintName(rr.Ptr)
}

type RP struct {
	Hdr  RR_Header
	Mbox string `dns:"domain-name"`
	Txt  string `dns:"domain-name"`
}

func (rr *RP) Header() *RR_Header { return &rr.Hdr }
func (rr *RP) copy() RR           { return &RP{*rr.Hdr.copyHeader(), rr.Mbox, rr.Txt} }
func (rr *RP) len() int           { return rr.Hdr.len() + len(rr.Mbox) + 1 + len(rr.Txt) + 1 }

func (rr *RP) String() string {
	return rr.Hdr.String() + rr.Mbox + " " + sprintTxt([]string{rr.Txt})
}

type SOA struct {
	Hdr     RR_Header
	Ns      string `dns:"cdomain-name"`
	Mbox    string `dns:"cdomain-name"`
	Serial  uint32
	Refresh uint32
	Retry   uint32
	Expire  uint32
	Minttl  uint32
}

func (rr *SOA) Header() *RR_Header { return &rr.Hdr }
func (rr *SOA) copy() RR {
	return &SOA{*rr.Hdr.copyHeader(), rr.Ns, rr.Mbox, rr.Serial, rr.Refresh, rr.Retry, rr.Expire, rr.Minttl}
}

func (rr *SOA) String() string {
	return rr.Hdr.String() + sprintName(rr.Ns) + " " + sprintName(rr.Mbox) +
		" " + strconv.FormatInt(int64(rr.Serial), 10) +
		" " + strconv.FormatInt(int64(rr.Refresh), 10) +
		" " + strconv.FormatInt(int64(rr.Retry), 10) +
		" " + strconv.FormatInt(int64(rr.Expire), 10) +
		" " + strconv.FormatInt(int64(rr.Minttl), 10)
}

func (rr *SOA) len() int {
	l := len(rr.Ns) + 1
	n := len(rr.Mbox) + 1
	return rr.Hdr.len() + l + n + 20
}

type TXT struct {
	Hdr RR_Header
	Txt []string `dns:"txt"`
}

func (rr *TXT) Header() *RR_Header { return &rr.Hdr }
func (rr *TXT) copy() RR {
	cp := make([]string, len(rr.Txt), cap(rr.Txt))
	copy(cp, rr.Txt)
	return &TXT{*rr.Hdr.copyHeader(), cp}
}

func (rr *TXT) String() string { return rr.Hdr.String() + sprintTxt(rr.Txt) }

func sprintName(s string) string {
	src := []byte(s)
	dst := make([]byte, 0, len(src))
	for i := 0; i < len(src); {
		if i+1 < len(src) && src[i] == '\\' && src[i+1] == '.' {
			dst = append(dst, src[i:i+2]...)
			i += 2
		} else {
			b, n := nextByte(src, i)
			if n == 0 {
				i++ // dangling back slash
			} else if b == '.' {
				dst = append(dst, b)
			} else {
				dst = appendDomainNameByte(dst, b)
			}
			i += n
		}
	}
	return string(dst)
}

func sprintCAAValue(s string) string {
	src := []byte(s)
	dst := make([]byte, 0, len(src))
	dst = append(dst, '"')
	for i := 0; i < len(src); {
		if i+1 < len(src) && src[i] == '\\' && src[i+1] == '.' {
			dst = append(dst, src[i:i+2]...)
			i += 2
		} else {
			b, n := nextByte(src, i)
			if n == 0 {
				i++ // dangling back slash
			} else if b == '.' {
				dst = append(dst, b)
			} else {
				if b < ' ' || b > '~' {
					dst = appendByte(dst, b)
				} else {
					dst = append(dst, b)
				}
			}
			i += n
		}
	}
	dst = append(dst, '"')
	return string(dst)
}

func sprintTxt(txt []string) string {
	var out []byte
	for i, s := range txt {
		if i > 0 {
			out = append(out, ` "`...)
		} else {
			out = append(out, '"')
		}
		bs := []byte(s)
		for j := 0; j < len(bs); {
			b, n := nextByte(bs, j)
			if n == 0 {
				break
			}
			out = appendTXTStringByte(out, b)
			j += n
		}
		out = append(out, '"')
	}
	return string(out)
}

func appendDomainNameByte(s []byte, b byte) []byte {
	switch b {
	case '.', ' ', '\'', '@', ';', '(', ')': // additional chars to escape
		return append(s, '\\', b)
	}
	return appendTXTStringByte(s, b)
}

func appendTXTStringByte(s []byte, b byte) []byte {
	switch b {
	case '\t':
		return append(s, '\\', 't')
	case '\r':
		return append(s, '\\', 'r')
	case '\n':
		return append(s, '\\', 'n')
	case '"', '\\':
		return append(s, '\\', b)
	}
	if b < ' ' || b > '~' {
		return appendByte(s, b)
	}
	return append(s, b)
}

func appendByte(s []byte, b byte) []byte {
	var buf [3]byte
	bufs := strconv.AppendInt(buf[:0], int64(b), 10)
	s = append(s, '\\')
	for i := 0; i < 3-len(bufs); i++ {
		s = append(s, '0')
	}
	for _, r := range bufs {
		s = append(s, r)
	}
	return s
}

func nextByte(b []byte, offset int) (byte, int) {
	if offset >= len(b) {
		return 0, 0
	}
	if b[offset] != '\\' {
		// not an escape sequence
		return b[offset], 1
	}
	switch len(b) - offset {
	case 1: // dangling escape
		return 0, 0
	case 2, 3: // too short to be \ddd
	default: // maybe \ddd
		if isDigit(b[offset+1]) && isDigit(b[offset+2]) && isDigit(b[offset+3]) {
			return dddToByte(b[offset+1:]), 4
		}
	}
	// not \ddd, maybe a control char
	switch b[offset+1] {
	case 't':
		return '\t', 2
	case 'r':
		return '\r', 2
	case 'n':
		return '\n', 2
	default:
		return b[offset+1], 2
	}
}

func (rr *TXT) len() int {
	l := rr.Hdr.len()
	for _, t := range rr.Txt {
		l += len(t) + 1
	}
	return l
}

type SPF struct {
	Hdr RR_Header
	Txt []string `dns:"txt"`
}

func (rr *SPF) Header() *RR_Header { return &rr.Hdr }
func (rr *SPF) copy() RR {
	cp := make([]string, len(rr.Txt), cap(rr.Txt))
	copy(cp, rr.Txt)
	return &SPF{*rr.Hdr.copyHeader(), cp}
}

func (rr *SPF) String() string { return rr.Hdr.String() + sprintTxt(rr.Txt) }

func (rr *SPF) len() int {
	l := rr.Hdr.len()
	for _, t := range rr.Txt {
		l += len(t) + 1
	}
	return l
}

type SRV struct {
	Hdr      RR_Header
	Priority uint16
	Weight   uint16
	Port     uint16
	Target   string `dns:"domain-name"`
}

func (rr *SRV) Header() *RR_Header { return &rr.Hdr }
func (rr *SRV) len() int           { l := len(rr.Target) + 1; return rr.Hdr.len() + l + 6 }
func (rr *SRV) copy() RR {
	return &SRV{*rr.Hdr.copyHeader(), rr.Priority, rr.Weight, rr.Port, rr.Target}
}

func (rr *SRV) String() string {
	return rr.Hdr.String() +
		strconv.Itoa(int(rr.Priority)) + " " +
		strconv.Itoa(int(rr.Weight)) + " " +
		strconv.Itoa(int(rr.Port)) + " " + sprintName(rr.Target)
}

type NAPTR struct {
	Hdr         RR_Header
	Order       uint16
	Preference  uint16
	Flags       string
	Service     string
	Regexp      string
	Replacement string `dns:"domain-name"`
}

func (rr *NAPTR) Header() *RR_Header { return &rr.Hdr }
func (rr *NAPTR) copy() RR {
	return &NAPTR{*rr.Hdr.copyHeader(), rr.Order, rr.Preference, rr.Flags, rr.Service, rr.Regexp, rr.Replacement}
}

func (rr *NAPTR) String() string {
	return rr.Hdr.String() +
		strconv.Itoa(int(rr.Order)) + " " +
		strconv.Itoa(int(rr.Preference)) + " " +
		"\"" + rr.Flags + "\" " +
		"\"" + rr.Service + "\" " +
		"\"" + rr.Regexp + "\" " +
		rr.Replacement
}

func (rr *NAPTR) len() int {
	return rr.Hdr.len() + 4 + len(rr.Flags) + 1 + len(rr.Service) + 1 +
		len(rr.Regexp) + 1 + len(rr.Replacement) + 1
}

// See RFC 4398.
type CERT struct {
	Hdr         RR_Header
	Type        uint16
	KeyTag      uint16
	Algorithm   uint8
	Certificate string `dns:"base64"`
}

func (rr *CERT) Header() *RR_Header { return &rr.Hdr }
func (rr *CERT) copy() RR {
	return &CERT{*rr.Hdr.copyHeader(), rr.Type, rr.KeyTag, rr.Algorithm, rr.Certificate}
}

func (rr *CERT) String() string {
	var (
		ok                  bool
		certtype, algorithm string
	)
	if certtype, ok = CertTypeToString[rr.Type]; !ok {
		certtype = strconv.Itoa(int(rr.Type))
	}
	if algorithm, ok = AlgorithmToString[rr.Algorithm]; !ok {
		algorithm = strconv.Itoa(int(rr.Algorithm))
	}
	return rr.Hdr.String() + certtype +
		" " + strconv.Itoa(int(rr.KeyTag)) +
		" " + algorithm +
		" " + rr.Certificate
}

func (rr *CERT) len() int {
	return rr.Hdr.len() + 5 +
		base64.StdEncoding.DecodedLen(len(rr.Certificate))
}

// See RFC 2672.
type DNAME struct {
	Hdr    RR_Header
	Target string `dns:"domain-name"`
}

func (rr *DNAME) Header() *RR_Header { return &rr.Hdr }
func (rr *DNAME) copy() RR           { return &DNAME{*rr.Hdr.copyHeader(), rr.Target} }
func (rr *DNAME) len() int           { l := len(rr.Target) + 1; return rr.Hdr.len() + l }

func (rr *DNAME) String() string {
	return rr.Hdr.String() + sprintName(rr.Target)
}

type A struct {
	Hdr RR_Header
	A   net.IP `dns:"a"`
}

func (rr *A) Header() *RR_Header { return &rr.Hdr }
func (rr *A) copy() RR           { return &A{*rr.Hdr.copyHeader(), copyIP(rr.A)} }
func (rr *A) len() int           { return rr.Hdr.len() + net.IPv4len }

func (rr *A) String() string {
	if rr.A == nil {
		return rr.Hdr.String()
	}
	return rr.Hdr.String() + rr.A.String()
}

type AAAA struct {
	Hdr  RR_Header
	AAAA net.IP `dns:"aaaa"`
}

func (rr *AAAA) Header() *RR_Header { return &rr.Hdr }
func (rr *AAAA) copy() RR           { return &AAAA{*rr.Hdr.copyHeader(), copyIP(rr.AAAA)} }
func (rr *AAAA) len() int           { return rr.Hdr.len() + net.IPv6len }

func (rr *AAAA) String() string {
	if rr.AAAA == nil {
		return rr.Hdr.String()
	}
	return rr.Hdr.String() + rr.AAAA.String()
}

type PX struct {
	Hdr        RR_Header
	Preference uint16
	Map822     string `dns:"domain-name"`
	Mapx400    string `dns:"domain-name"`
}

func (rr *PX) Header() *RR_Header { return &rr.Hdr }
func (rr *PX) copy() RR           { return &PX{*rr.Hdr.copyHeader(), rr.Preference, rr.Map822, rr.Mapx400} }
func (rr *PX) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Map822) + " " + sprintName(rr.Mapx400)
}
func (rr *PX) len() int { return rr.Hdr.len() + 2 + len(rr.Map822) + 1 + len(rr.Mapx400) + 1 }

type GPOS struct {
	Hdr       RR_Header
	Longitude string
	Latitude  string
	Altitude  string
}

func (rr *GPOS) Header() *RR_Header { return &rr.Hdr }
func (rr *GPOS) copy() RR           { return &GPOS{*rr.Hdr.copyHeader(), rr.Longitude, rr.Latitude, rr.Altitude} }
func (rr *GPOS) len() int {
	return rr.Hdr.len() + len(rr.Longitude) + len(rr.Latitude) + len(rr.Altitude) + 3
}
func (rr *GPOS) String() string {
	return rr.Hdr.String() + rr.Longitude + " " + rr.Latitude + " " + rr.Altitude
}

type LOC struct {
	Hdr       RR_Header
	Version   uint8
	Size      uint8
	HorizPre  uint8
	VertPre   uint8
	Latitude  uint32
	Longitude uint32
	Altitude  uint32
}

func (rr *LOC) Header() *RR_Header { return &rr.Hdr }
func (rr *LOC) len() int           { return rr.Hdr.len() + 4 + 12 }
func (rr *LOC) copy() RR {
	return &LOC{*rr.Hdr.copyHeader(), rr.Version, rr.Size, rr.HorizPre, rr.VertPre, rr.Latitude, rr.Longitude, rr.Altitude}
}

// cmToM takes a cm value expressed in RFC1876 SIZE mantissa/exponent
// format and returns a string in m (two decimals for the cm)
func cmToM(m, e uint8) string {
	if e < 2 {
		if e == 1 {
			m *= 10
		}

		return fmt.Sprintf("0.%02d", m)
	}

	s := fmt.Sprintf("%d", m)
	for e > 2 {
		s += "0"
		e--
	}
	return s
}

// String returns a string version of a LOC
func (rr *LOC) String() string {
	s := rr.Hdr.String()

	lat := rr.Latitude
	ns := "N"
	if lat > LOC_EQUATOR {
		lat = lat - LOC_EQUATOR
	} else {
		ns = "S"
		lat = LOC_EQUATOR - lat
	}
	h := lat / LOC_DEGREES
	lat = lat % LOC_DEGREES
	m := lat / LOC_HOURS
	lat = lat % LOC_HOURS
	s += fmt.Sprintf("%02d %02d %0.3f %s ", h, m, (float64(lat) / 1000), ns)

	lon := rr.Longitude
	ew := "E"
	if lon > LOC_PRIMEMERIDIAN {
		lon = lon - LOC_PRIMEMERIDIAN
	} else {
		ew = "W"
		lon = LOC_PRIMEMERIDIAN - lon
	}
	h = lon / LOC_DEGREES
	lon = lon % LOC_DEGREES
	m = lon / LOC_HOURS
	lon = lon % LOC_HOURS
	s += fmt.Sprintf("%02d %02d %0.3f %s ", h, m, (float64(lon) / 1000), ew)

	var alt = float64(rr.Altitude) / 100
	alt -= LOC_ALTITUDEBASE
	if rr.Altitude%100 != 0 {
		s += fmt.Sprintf("%.2fm ", alt)
	} else {
		s += fmt.Sprintf("%.0fm ", alt)
	}

	s += cmToM((rr.Size&0xf0)>>4, rr.Size&0x0f) + "m "
	s += cmToM((rr.HorizPre&0xf0)>>4, rr.HorizPre&0x0f) + "m "
	s += cmToM((rr.VertPre&0xf0)>>4, rr.VertPre&0x0f) + "m"

	return s
}

// SIG is identical to RRSIG and nowadays only used for SIG(0), RFC2931.
type SIG struct {
	RRSIG
}

type RRSIG struct {
	Hdr         RR_Header
	TypeCovered uint16
	Algorithm   uint8
	Labels      uint8
	OrigTtl     uint32
	Expiration  uint32
	Inception   uint32
	KeyTag      uint16
	SignerName  string `dns:"domain-name"`
	Signature   string `dns:"base64"`
}

func (rr *RRSIG) Header() *RR_Header { return &rr.Hdr }
func (rr *RRSIG) copy() RR {
	return &RRSIG{*rr.Hdr.copyHeader(), rr.TypeCovered, rr.Algorithm, rr.Labels, rr.OrigTtl, rr.Expiration, rr.Inception, rr.KeyTag, rr.SignerName, rr.Signature}
}

func (rr *RRSIG) String() string {
	s := rr.Hdr.String()
	s += Type(rr.TypeCovered).String()
	s += " " + strconv.Itoa(int(rr.Algorithm)) +
		" " + strconv.Itoa(int(rr.Labels)) +
		" " + strconv.FormatInt(int64(rr.OrigTtl), 10) +
		" " + TimeToString(rr.Expiration) +
		" " + TimeToString(rr.Inception) +
		" " + strconv.Itoa(int(rr.KeyTag)) +
		" " + sprintName(rr.SignerName) +
		" " + rr.Signature
	return s
}

func (rr *RRSIG) len() int {
	return rr.Hdr.len() + len(rr.SignerName) + 1 +
		base64.StdEncoding.DecodedLen(len(rr.Signature)) + 18
}

type NSEC struct {
	Hdr        RR_Header
	NextDomain string   `dns:"domain-name"`
	TypeBitMap []uint16 `dns:"nsec"`
}

func (rr *NSEC) Header() *RR_Header { return &rr.Hdr }
func (rr *NSEC) copy() RR {
	cp := make([]uint16, len(rr.TypeBitMap), cap(rr.TypeBitMap))
	copy(cp, rr.TypeBitMap)
	return &NSEC{*rr.Hdr.copyHeader(), rr.NextDomain, cp}
}

func (rr *NSEC) String() string {
	s := rr.Hdr.String() + sprintName(rr.NextDomain)
	for i := 0; i < len(rr.TypeBitMap); i++ {
		s += " " + Type(rr.TypeBitMap[i]).String()
	}
	return s
}

func (rr *NSEC) len() int {
	l := rr.Hdr.len() + len(rr.NextDomain) + 1
	lastwindow := uint32(2 ^ 32 + 1)
	for _, t := range rr.TypeBitMap {
		window := t / 256
		if uint32(window) != lastwindow {
			l += 1 + 32
		}
		lastwindow = uint32(window)
	}
	return l
}

type DLV struct {
	DS
}

type CDS struct {
	DS
}

type DS struct {
	Hdr        RR_Header
	KeyTag     uint16
	Algorithm  uint8
	DigestType uint8
	Digest     string `dns:"hex"`
}

func (rr *DS) Header() *RR_Header { return &rr.Hdr }
func (rr *DS) len() int           { return rr.Hdr.len() + 4 + len(rr.Digest)/2 }
func (rr *DS) copy() RR {
	return &DS{*rr.Hdr.copyHeader(), rr.KeyTag, rr.Algorithm, rr.DigestType, rr.Digest}
}

func (rr *DS) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.KeyTag)) +
		" " + strconv.Itoa(int(rr.Algorithm)) +
		" " + strconv.Itoa(int(rr.DigestType)) +
		" " + strings.ToUpper(rr.Digest)
}

type KX struct {
	Hdr        RR_Header
	Preference uint16
	Exchanger  string `dns:"domain-name"`
}

func (rr *KX) Header() *RR_Header { return &rr.Hdr }
func (rr *KX) len() int           { return rr.Hdr.len() + 2 + len(rr.Exchanger) + 1 }
func (rr *KX) copy() RR           { return &KX{*rr.Hdr.copyHeader(), rr.Preference, rr.Exchanger} }

func (rr *KX) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) +
		" " + sprintName(rr.Exchanger)
}

type TA struct {
	Hdr        RR_Header
	KeyTag     uint16
	Algorithm  uint8
	DigestType uint8
	Digest     string `dns:"hex"`
}

func (rr *TA) Header() *RR_Header { return &rr.Hdr }
func (rr *TA) len() int           { return rr.Hdr.len() + 4 + len(rr.Digest)/2 }
func (rr *TA) copy() RR {
	return &TA{*rr.Hdr.copyHeader(), rr.KeyTag, rr.Algorithm, rr.DigestType, rr.Digest}
}

func (rr *TA) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.KeyTag)) +
		" " + strconv.Itoa(int(rr.Algorithm)) +
		" " + strconv.Itoa(int(rr.DigestType)) +
		" " + strings.ToUpper(rr.Digest)
}

type TALINK struct {
	Hdr          RR_Header
	PreviousName string `dns:"domain-name"`
	NextName     string `dns:"domain-name"`
}

func (rr *TALINK) Header() *RR_Header { return &rr.Hdr }
func (rr *TALINK) copy() RR           { return &TALINK{*rr.Hdr.copyHeader(), rr.PreviousName, rr.NextName} }
func (rr *TALINK) len() int           { return rr.Hdr.len() + len(rr.PreviousName) + len(rr.NextName) + 2 }

func (rr *TALINK) String() string {
	return rr.Hdr.String() +
		sprintName(rr.PreviousName) + " " + sprintName(rr.NextName)
}

type SSHFP struct {
	Hdr         RR_Header
	Algorithm   uint8
	Type        uint8
	FingerPrint string `dns:"hex"`
}

func (rr *SSHFP) Header() *RR_Header { return &rr.Hdr }
func (rr *SSHFP) len() int           { return rr.Hdr.len() + 2 + len(rr.FingerPrint)/2 }
func (rr *SSHFP) copy() RR {
	return &SSHFP{*rr.Hdr.copyHeader(), rr.Algorithm, rr.Type, rr.FingerPrint}
}

func (rr *SSHFP) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Algorithm)) +
		" " + strconv.Itoa(int(rr.Type)) +
		" " + strings.ToUpper(rr.FingerPrint)
}

type IPSECKEY struct {
	Hdr        RR_Header
	Precedence uint8
	// GatewayType: 1: A record, 2: AAAA record, 3: domainname.
	// 0 is use for no type and GatewayName should be "." then.
	GatewayType uint8
	Algorithm   uint8
	// Gateway can be an A record, AAAA record or a domain name.
	GatewayA    net.IP `dns:"a"`
	GatewayAAAA net.IP `dns:"aaaa"`
	GatewayName string `dns:"domain-name"`
	PublicKey   string `dns:"base64"`
}

func (rr *IPSECKEY) Header() *RR_Header { return &rr.Hdr }
func (rr *IPSECKEY) copy() RR {
	return &IPSECKEY{*rr.Hdr.copyHeader(), rr.Precedence, rr.GatewayType, rr.Algorithm, rr.GatewayA, rr.GatewayAAAA, rr.GatewayName, rr.PublicKey}
}

func (rr *IPSECKEY) String() string {
	s := rr.Hdr.String() + strconv.Itoa(int(rr.Precedence)) +
		" " + strconv.Itoa(int(rr.GatewayType)) +
		" " + strconv.Itoa(int(rr.Algorithm))
	switch rr.GatewayType {
	case 0:
		fallthrough
	case 3:
		s += " " + rr.GatewayName
	case 1:
		s += " " + rr.GatewayA.String()
	case 2:
		s += " " + rr.GatewayAAAA.String()
	default:
		s += " ."
	}
	s += " " + rr.PublicKey
	return s
}

func (rr *IPSECKEY) len() int {
	l := rr.Hdr.len() + 3 + 1
	switch rr.GatewayType {
	default:
		fallthrough
	case 0:
		fallthrough
	case 3:
		l += len(rr.GatewayName)
	case 1:
		l += 4
	case 2:
		l += 16
	}
	return l + base64.StdEncoding.DecodedLen(len(rr.PublicKey))
}

type KEY struct {
	DNSKEY
}

type CDNSKEY struct {
	DNSKEY
}

type DNSKEY struct {
	Hdr       RR_Header
	Flags     uint16
	Protocol  uint8
	Algorithm uint8
	PublicKey string `dns:"base64"`
}

func (rr *DNSKEY) Header() *RR_Header { return &rr.Hdr }
func (rr *DNSKEY) len() int {
	return rr.Hdr.len() + 4 + base64.StdEncoding.DecodedLen(len(rr.PublicKey))
}
func (rr *DNSKEY) copy() RR {
	return &DNSKEY{*rr.Hdr.copyHeader(), rr.Flags, rr.Protocol, rr.Algorithm, rr.PublicKey}
}

func (rr *DNSKEY) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Flags)) +
		" " + strconv.Itoa(int(rr.Protocol)) +
		" " + strconv.Itoa(int(rr.Algorithm)) +
		" " + rr.PublicKey
}

type RKEY struct {
	Hdr       RR_Header
	Flags     uint16
	Protocol  uint8
	Algorithm uint8
	PublicKey string `dns:"base64"`
}

func (rr *RKEY) Header() *RR_Header { return &rr.Hdr }
func (rr *RKEY) len() int           { return rr.Hdr.len() + 4 + base64.StdEncoding.DecodedLen(len(rr.PublicKey)) }
func (rr *RKEY) copy() RR {
	return &RKEY{*rr.Hdr.copyHeader(), rr.Flags, rr.Protocol, rr.Algorithm, rr.PublicKey}
}

func (rr *RKEY) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Flags)) +
		" " + strconv.Itoa(int(rr.Protocol)) +
		" " + strconv.Itoa(int(rr.Algorithm)) +
		" " + rr.PublicKey
}

type NSAP struct {
	Hdr  RR_Header
	Nsap string
}

func (rr *NSAP) Header() *RR_Header { return &rr.Hdr }
func (rr *NSAP) copy() RR           { return &NSAP{*rr.Hdr.copyHeader(), rr.Nsap} }
func (rr *NSAP) String() string     { return rr.Hdr.String() + "0x" + rr.Nsap }
func (rr *NSAP) len() int           { return rr.Hdr.len() + 1 + len(rr.Nsap) + 1 }

type NSAPPTR struct {
	Hdr RR_Header
	Ptr string `dns:"domain-name"`
}

func (rr *NSAPPTR) Header() *RR_Header { return &rr.Hdr }
func (rr *NSAPPTR) copy() RR           { return &NSAPPTR{*rr.Hdr.copyHeader(), rr.Ptr} }
func (rr *NSAPPTR) String() string     { return rr.Hdr.String() + sprintName(rr.Ptr) }
func (rr *NSAPPTR) len() int           { return rr.Hdr.len() + len(rr.Ptr) }

type NSEC3 struct {
	Hdr        RR_Header
	Hash       uint8
	Flags      uint8
	Iterations uint16
	SaltLength uint8
	Salt       string `dns:"size-hex"`
	HashLength uint8
	NextDomain string   `dns:"size-base32"`
	TypeBitMap []uint16 `dns:"nsec"`
}

func (rr *NSEC3) Header() *RR_Header { return &rr.Hdr }
func (rr *NSEC3) copy() RR {
	cp := make([]uint16, len(rr.TypeBitMap), cap(rr.TypeBitMap))
	copy(cp, rr.TypeBitMap)
	return &NSEC3{*rr.Hdr.copyHeader(), rr.Hash, rr.Flags, rr.Iterations, rr.SaltLength, rr.Salt, rr.HashLength, rr.NextDomain, cp}
}

func (rr *NSEC3) String() string {
	s := rr.Hdr.String()
	s += strconv.Itoa(int(rr.Hash)) +
		" " + strconv.Itoa(int(rr.Flags)) +
		" " + strconv.Itoa(int(rr.Iterations)) +
		" " + saltToString(rr.Salt) +
		" " + rr.NextDomain
	for i := 0; i < len(rr.TypeBitMap); i++ {
		s += " " + Type(rr.TypeBitMap[i]).String()
	}
	return s
}

func (rr *NSEC3) len() int {
	l := rr.Hdr.len() + 6 + len(rr.Salt)/2 + 1 + len(rr.NextDomain) + 1
	lastwindow := uint32(2 ^ 32 + 1)
	for _, t := range rr.TypeBitMap {
		window := t / 256
		if uint32(window) != lastwindow {
			l += 1 + 32
		}
		lastwindow = uint32(window)
	}
	return l
}

type NSEC3PARAM struct {
	Hdr        RR_Header
	Hash       uint8
	Flags      uint8
	Iterations uint16
	SaltLength uint8
	Salt       string `dns:"hex"`
}

func (rr *NSEC3PARAM) Header() *RR_Header { return &rr.Hdr }
func (rr *NSEC3PARAM) len() int           { return rr.Hdr.len() + 2 + 4 + 1 + len(rr.Salt)/2 }
func (rr *NSEC3PARAM) copy() RR {
	return &NSEC3PARAM{*rr.Hdr.copyHeader(), rr.Hash, rr.Flags, rr.Iterations, rr.SaltLength, rr.Salt}
}

func (rr *NSEC3PARAM) String() string {
	s := rr.Hdr.String()
	s += strconv.Itoa(int(rr.Hash)) +
		" " + strconv.Itoa(int(rr.Flags)) +
		" " + strconv.Itoa(int(rr.Iterations)) +
		" " + saltToString(rr.Salt)
	return s
}

type TKEY struct {
	Hdr        RR_Header
	Algorithm  string `dns:"domain-name"`
	Inception  uint32
	Expiration uint32
	Mode       uint16
	Error      uint16
	KeySize    uint16
	Key        string
	OtherLen   uint16
	OtherData  string
}

func (rr *TKEY) Header() *RR_Header { return &rr.Hdr }
func (rr *TKEY) copy() RR {
	return &TKEY{*rr.Hdr.copyHeader(), rr.Algorithm, rr.Inception, rr.Expiration, rr.Mode, rr.Error, rr.KeySize, rr.Key, rr.OtherLen, rr.OtherData}
}

func (rr *TKEY) String() string {
	// It has no presentation format
	return ""
}

func (rr *TKEY) len() int {
	return rr.Hdr.len() + len(rr.Algorithm) + 1 + 4 + 4 + 6 +
		len(rr.Key) + 2 + len(rr.OtherData)
}

// RFC3597 represents an unknown/generic RR.
type RFC3597 struct {
	Hdr   RR_Header
	Rdata string `dns:"hex"`
}

func (rr *RFC3597) Header() *RR_Header { return &rr.Hdr }
func (rr *RFC3597) copy() RR           { return &RFC3597{*rr.Hdr.copyHeader(), rr.Rdata} }
func (rr *RFC3597) len() int           { return rr.Hdr.len() + len(rr.Rdata)/2 + 2 }

func (rr *RFC3597) String() string {
	// Let's call it a hack
	s := rfc3597Header(rr.Hdr)

	s += "\\# " + strconv.Itoa(len(rr.Rdata)/2) + " " + rr.Rdata
	return s
}

func rfc3597Header(h RR_Header) string {
	var s string

	s += sprintName(h.Name) + "\t"
	s += strconv.FormatInt(int64(h.Ttl), 10) + "\t"
	s += "CLASS" + strconv.Itoa(int(h.Class)) + "\t"
	s += "TYPE" + strconv.Itoa(int(h.Rrtype)) + "\t"
	return s
}

type URI struct {
	Hdr      RR_Header
	Priority uint16
	Weight   uint16
	Target   []string `dns:"txt"`
}

func (rr *URI) Header() *RR_Header { return &rr.Hdr }
func (rr *URI) copy() RR {
	cp := make([]string, len(rr.Target), cap(rr.Target))
	copy(cp, rr.Target)
	return &URI{*rr.Hdr.copyHeader(), rr.Weight, rr.Priority, cp}
}

func (rr *URI) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Priority)) +
		" " + strconv.Itoa(int(rr.Weight)) + sprintTxt(rr.Target)
}

func (rr *URI) len() int {
	l := rr.Hdr.len() + 4
	for _, t := range rr.Target {
		l += len(t) + 1
	}
	return l
}

type DHCID struct {
	Hdr    RR_Header
	Digest string `dns:"base64"`
}

func (rr *DHCID) Header() *RR_Header { return &rr.Hdr }
func (rr *DHCID) copy() RR           { return &DHCID{*rr.Hdr.copyHeader(), rr.Digest} }
func (rr *DHCID) String() string     { return rr.Hdr.String() + rr.Digest }
func (rr *DHCID) len() int           { return rr.Hdr.len() + base64.StdEncoding.DecodedLen(len(rr.Digest)) }

type TLSA struct {
	Hdr          RR_Header
	Usage        uint8
	Selector     uint8
	MatchingType uint8
	Certificate  string `dns:"hex"`
}

func (rr *TLSA) Header() *RR_Header { return &rr.Hdr }
func (rr *TLSA) len() int           { return rr.Hdr.len() + 3 + len(rr.Certificate)/2 }

func (rr *TLSA) copy() RR {
	return &TLSA{*rr.Hdr.copyHeader(), rr.Usage, rr.Selector, rr.MatchingType, rr.Certificate}
}

func (rr *TLSA) String() string {
	return rr.Hdr.String() +
		strconv.Itoa(int(rr.Usage)) +
		" " + strconv.Itoa(int(rr.Selector)) +
		" " + strconv.Itoa(int(rr.MatchingType)) +
		" " + rr.Certificate
}

type HIP struct {
	Hdr                RR_Header
	HitLength          uint8
	PublicKeyAlgorithm uint8
	PublicKeyLength    uint16
	Hit                string   `dns:"hex"`
	PublicKey          string   `dns:"base64"`
	RendezvousServers  []string `dns:"domain-name"`
}

func (rr *HIP) Header() *RR_Header { return &rr.Hdr }
func (rr *HIP) copy() RR {
	cp := make([]string, len(rr.RendezvousServers), cap(rr.RendezvousServers))
	copy(cp, rr.RendezvousServers)
	return &HIP{*rr.Hdr.copyHeader(), rr.HitLength, rr.PublicKeyAlgorithm, rr.PublicKeyLength, rr.Hit, rr.PublicKey, cp}
}

func (rr *HIP) String() string {
	s := rr.Hdr.String() +
		strconv.Itoa(int(rr.PublicKeyAlgorithm)) +
		" " + rr.Hit +
		" " + rr.PublicKey
	for _, d := range rr.RendezvousServers {
		s += " " + sprintName(d)
	}
	return s
}

func (rr *HIP) len() int {
	l := rr.Hdr.len() + 4 +
		len(rr.Hit)/2 +
		base64.StdEncoding.DecodedLen(len(rr.PublicKey))
	for _, d := range rr.RendezvousServers {
		l += len(d) + 1
	}
	return l
}

type NINFO struct {
	Hdr    RR_Header
	ZSData []string `dns:"txt"`
}

func (rr *NINFO) Header() *RR_Header { return &rr.Hdr }
func (rr *NINFO) copy() RR {
	cp := make([]string, len(rr.ZSData), cap(rr.ZSData))
	copy(cp, rr.ZSData)
	return &NINFO{*rr.Hdr.copyHeader(), cp}
}

func (rr *NINFO) String() string { return rr.Hdr.String() + sprintTxt(rr.ZSData) }

func (rr *NINFO) len() int {
	l := rr.Hdr.len()
	for _, t := range rr.ZSData {
		l += len(t) + 1
	}
	return l
}

type WKS struct {
	Hdr      RR_Header
	Address  net.IP `dns:"a"`
	Protocol uint8
	BitMap   []uint16 `dns:"wks"`
}

func (rr *WKS) Header() *RR_Header { return &rr.Hdr }
func (rr *WKS) len() int           { return rr.Hdr.len() + net.IPv4len + 1 }

func (rr *WKS) copy() RR {
	cp := make([]uint16, len(rr.BitMap), cap(rr.BitMap))
	copy(cp, rr.BitMap)
	return &WKS{*rr.Hdr.copyHeader(), copyIP(rr.Address), rr.Protocol, cp}
}

func (rr *WKS) String() (s string) {
	s = rr.Hdr.String()
	if rr.Address != nil {
		s += rr.Address.String()
	}
	// TODO(miek): missing protocol here, see /etc/protocols
	for i := 0; i < len(rr.BitMap); i++ {
		// should lookup the port
		s += " " + strconv.Itoa(int(rr.BitMap[i]))
	}
	return s
}

type NID struct {
	Hdr        RR_Header
	Preference uint16
	NodeID     uint64
}

func (rr *NID) Header() *RR_Header { return &rr.Hdr }
func (rr *NID) copy() RR           { return &NID{*rr.Hdr.copyHeader(), rr.Preference, rr.NodeID} }
func (rr *NID) len() int           { return rr.Hdr.len() + 2 + 8 }

func (rr *NID) String() string {
	s := rr.Hdr.String() + strconv.Itoa(int(rr.Preference))
	node := fmt.Sprintf("%0.16x", rr.NodeID)
	s += " " + node[0:4] + ":" + node[4:8] + ":" + node[8:12] + ":" + node[12:16]
	return s
}

type L32 struct {
	Hdr        RR_Header
	Preference uint16
	Locator32  net.IP `dns:"a"`
}

func (rr *L32) Header() *RR_Header { return &rr.Hdr }
func (rr *L32) copy() RR           { return &L32{*rr.Hdr.copyHeader(), rr.Preference, copyIP(rr.Locator32)} }
func (rr *L32) len() int           { return rr.Hdr.len() + net.IPv4len }

func (rr *L32) String() string {
	if rr.Locator32 == nil {
		return rr.Hdr.String() + strconv.Itoa(int(rr.Preference))
	}
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) +
		" " + rr.Locator32.String()
}

type L64 struct {
	Hdr        RR_Header
	Preference uint16
	Locator64  uint64
}

func (rr *L64) Header() *RR_Header { return &rr.Hdr }
func (rr *L64) copy() RR           { return &L64{*rr.Hdr.copyHeader(), rr.Preference, rr.Locator64} }
func (rr *L64) len() int           { return rr.Hdr.len() + 2 + 8 }

func (rr *L64) String() string {
	s := rr.Hdr.String() + strconv.Itoa(int(rr.Preference))
	node := fmt.Sprintf("%0.16X", rr.Locator64)
	s += " " + node[0:4] + ":" + node[4:8] + ":" + node[8:12] + ":" + node[12:16]
	return s
}

type LP struct {
	Hdr        RR_Header
	Preference uint16
	Fqdn       string `dns:"domain-name"`
}

func (rr *LP) Header() *RR_Header { return &rr.Hdr }
func (rr *LP) copy() RR           { return &LP{*rr.Hdr.copyHeader(), rr.Preference, rr.Fqdn} }
func (rr *LP) len() int           { return rr.Hdr.len() + 2 + len(rr.Fqdn) + 1 }

func (rr *LP) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Fqdn)
}

type EUI48 struct {
	Hdr     RR_Header
	Address uint64 `dns:"uint48"`
}

func (rr *EUI48) Header() *RR_Header { return &rr.Hdr }
func (rr *EUI48) copy() RR           { return &EUI48{*rr.Hdr.copyHeader(), rr.Address} }
func (rr *EUI48) String() string     { return rr.Hdr.String() + euiToString(rr.Address, 48) }
func (rr *EUI48) len() int           { return rr.Hdr.len() + 6 }

type EUI64 struct {
	Hdr     RR_Header
	Address uint64
}

func (rr *EUI64) Header() *RR_Header { return &rr.Hdr }
func (rr *EUI64) copy() RR           { return &EUI64{*rr.Hdr.copyHeader(), rr.Address} }
func (rr *EUI64) String() string     { return rr.Hdr.String() + euiToString(rr.Address, 64) }
func (rr *EUI64) len() int           { return rr.Hdr.len() + 8 }

type CAA struct {
	Hdr   RR_Header
	Flag  uint8
	Tag   string
	Value string `dns:"octet"`
}

func (rr *CAA) Header() *RR_Header { return &rr.Hdr }
func (rr *CAA) copy() RR           { return &CAA{*rr.Hdr.copyHeader(), rr.Flag, rr.Tag, rr.Value} }
func (rr *CAA) len() int           { return rr.Hdr.len() + 1 + len(rr.Tag) + len(rr.Value)/2 }
func (rr *CAA) String() string     { return rr.Hdr.String() + strconv.Itoa(int(rr.Flag)) + " " + rr.Tag + " " + sprintCAAValue(rr.Value) }


type UID struct {
	Hdr RR_Header
	Uid uint32
}

func (rr *UID) Header() *RR_Header { return &rr.Hdr }
func (rr *UID) copy() RR           { return &UID{*rr.Hdr.copyHeader(), rr.Uid} }
func (rr *UID) String() string     { return rr.Hdr.String() + strconv.FormatInt(int64(rr.Uid), 10) }
func (rr *UID) len() int           { return rr.Hdr.len() + 4 }

type GID struct {
	Hdr RR_Header
	Gid uint32
}

func (rr *GID) Header() *RR_Header { return &rr.Hdr }
func (rr *GID) copy() RR           { return &GID{*rr.Hdr.copyHeader(), rr.Gid} }
func (rr *GID) String() string     { return rr.Hdr.String() + strconv.FormatInt(int64(rr.Gid), 10) }
func (rr *GID) len() int           { return rr.Hdr.len() + 4 }

type UINFO struct {
	Hdr   RR_Header
	Uinfo string
}

func (rr *UINFO) Header() *RR_Header { return &rr.Hdr }
func (rr *UINFO) copy() RR           { return &UINFO{*rr.Hdr.copyHeader(), rr.Uinfo} }
func (rr *UINFO) String() string     { return rr.Hdr.String() + sprintTxt([]string{rr.Uinfo}) }
func (rr *UINFO) len() int           { return rr.Hdr.len() + len(rr.Uinfo) + 1 }

type EID struct {
	Hdr      RR_Header
	Endpoint string `dns:"hex"`
}

func (rr *EID) Header() *RR_Header { return &rr.Hdr }
func (rr *EID) copy() RR           { return &EID{*rr.Hdr.copyHeader(), rr.Endpoint} }
func (rr *EID) String() string     { return rr.Hdr.String() + strings.ToUpper(rr.Endpoint) }
func (rr *EID) len() int           { return rr.Hdr.len() + len(rr.Endpoint)/2 }

type NIMLOC struct {
	Hdr     RR_Header
	Locator string `dns:"hex"`
}

func (rr *NIMLOC) Header() *RR_Header { return &rr.Hdr }
func (rr *NIMLOC) copy() RR           { return &NIMLOC{*rr.Hdr.copyHeader(), rr.Locator} }
func (rr *NIMLOC) String() string     { return rr.Hdr.String() + strings.ToUpper(rr.Locator) }
func (rr *NIMLOC) len() int           { return rr.Hdr.len() + len(rr.Locator)/2 }

type OPENPGPKEY struct {
	Hdr       RR_Header
	PublicKey string `dns:"base64"`
}

func (rr *OPENPGPKEY) Header() *RR_Header { return &rr.Hdr }
func (rr *OPENPGPKEY) copy() RR           { return &OPENPGPKEY{*rr.Hdr.copyHeader(), rr.PublicKey} }
func (rr *OPENPGPKEY) String() string     { return rr.Hdr.String() + rr.PublicKey }
func (rr *OPENPGPKEY) len() int {
	return rr.Hdr.len() + base64.StdEncoding.DecodedLen(len(rr.PublicKey))
}

// TimeToString translates the RRSIG's incep. and expir. times to the
// string representation used when printing the record.
// It takes serial arithmetic (RFC 1982) into account.
func TimeToString(t uint32) string {
	mod := ((int64(t) - time.Now().Unix()) / year68) - 1
	if mod < 0 {
		mod = 0
	}
	ti := time.Unix(int64(t)-(mod*year68), 0).UTC()
	return ti.Format("20060102150405")
}

// StringToTime translates the RRSIG's incep. and expir. times from
// string values like "20110403154150" to an 32 bit integer.
// It takes serial arithmetic (RFC 1982) into account.
func StringToTime(s string) (uint32, error) {
	t, e := time.Parse("20060102150405", s)
	if e != nil {
		return 0, e
	}
	mod := (t.Unix() / year68) - 1
	if mod < 0 {
		mod = 0
	}
	return uint32(t.Unix() - (mod * year68)), nil
}

// saltToString converts a NSECX salt to uppercase and
// returns "-" when it is empty
func saltToString(s string) string {
	if len(s) == 0 {
		return "-"
	}
	return strings.ToUpper(s)
}

func euiToString(eui uint64, bits int) (hex string) {
	switch bits {
	case 64:
		hex = fmt.Sprintf("%16.16x", eui)
		hex = hex[0:2] + "-" + hex[2:4] + "-" + hex[4:6] + "-" + hex[6:8] +
			"-" + hex[8:10] + "-" + hex[10:12] + "-" + hex[12:14] + "-" + hex[14:16]
	case 48:
		hex = fmt.Sprintf("%12.12x", eui)
		hex = hex[0:2] + "-" + hex[2:4] + "-" + hex[4:6] + "-" + hex[6:8] +
			"-" + hex[8:10] + "-" + hex[10:12]
	}
	return
}

// copyIP returns a copy of ip.
func copyIP(ip net.IP) net.IP {
	p := make(net.IP, len(ip))
	copy(p, ip)
	return p
}

// Map of constructors for each RR type.
var typeToRR = map[uint16]func() RR{
	TypeA:          func() RR { return new(A) },
	TypeAAAA:       func() RR { return new(AAAA) },
	TypeAFSDB:      func() RR { return new(AFSDB) },
	TypeCAA:        func() RR { return new(CAA) },
	TypeCDS:        func() RR { return new(CDS) },
	TypeCERT:       func() RR { return new(CERT) },
	TypeCNAME:      func() RR { return new(CNAME) },
	TypeDHCID:      func() RR { return new(DHCID) },
	TypeDLV:        func() RR { return new(DLV) },
	TypeDNAME:      func() RR { return new(DNAME) },
	TypeKEY:        func() RR { return new(KEY) },
	TypeDNSKEY:     func() RR { return new(DNSKEY) },
	TypeDS:         func() RR { return new(DS) },
	TypeEUI48:      func() RR { return new(EUI48) },
	TypeEUI64:      func() RR { return new(EUI64) },
	TypeGID:        func() RR { return new(GID) },
	TypeGPOS:       func() RR { return new(GPOS) },
	TypeEID:        func() RR { return new(EID) },
	TypeHINFO:      func() RR { return new(HINFO) },
	TypeHIP:        func() RR { return new(HIP) },
	TypeIPSECKEY:   func() RR { return new(IPSECKEY) },
	TypeKX:         func() RR { return new(KX) },
	TypeL32:        func() RR { return new(L32) },
	TypeL64:        func() RR { return new(L64) },
	TypeLOC:        func() RR { return new(LOC) },
	TypeLP:         func() RR { return new(LP) },
	TypeMB:         func() RR { return new(MB) },
	TypeMD:         func() RR { return new(MD) },
	TypeMF:         func() RR { return new(MF) },
	TypeMG:         func() RR { return new(MG) },
	TypeMINFO:      func() RR { return new(MINFO) },
	TypeMR:         func() RR { return new(MR) },
	TypeMX:         func() RR { return new(MX) },
	TypeNAPTR:      func() RR { return new(NAPTR) },
	TypeNID:        func() RR { return new(NID) },
	TypeNINFO:      func() RR { return new(NINFO) },
	TypeNIMLOC:     func() RR { return new(NIMLOC) },
	TypeNS:         func() RR { return new(NS) },
	TypeNSAP:       func() RR { return new(NSAP) },
	TypeNSAPPTR:    func() RR { return new(NSAPPTR) },
	TypeNSEC3:      func() RR { return new(NSEC3) },
	TypeNSEC3PARAM: func() RR { return new(NSEC3PARAM) },
	TypeNSEC:       func() RR { return new(NSEC) },
	TypeOPENPGPKEY: func() RR { return new(OPENPGPKEY) },
	TypeOPT:        func() RR { return new(OPT) },
	TypePTR:        func() RR { return new(PTR) },
	TypeRKEY:       func() RR { return new(RKEY) },
	TypeRP:         func() RR { return new(RP) },
	TypePX:         func() RR { return new(PX) },
	TypeSIG:        func() RR { return new(SIG) },
	TypeRRSIG:      func() RR { return new(RRSIG) },
	TypeRT:         func() RR { return new(RT) },
	TypeSOA:        func() RR { return new(SOA) },
	TypeSPF:        func() RR { return new(SPF) },
	TypeSRV:        func() RR { return new(SRV) },
	TypeSSHFP:      func() RR { return new(SSHFP) },
	TypeTA:         func() RR { return new(TA) },
	TypeTALINK:     func() RR { return new(TALINK) },
	TypeTKEY:       func() RR { return new(TKEY) },
	TypeTLSA:       func() RR { return new(TLSA) },
	TypeTSIG:       func() RR { return new(TSIG) },
	TypeTXT:        func() RR { return new(TXT) },
	TypeUID:        func() RR { return new(UID) },
	TypeUINFO:      func() RR { return new(UINFO) },
	TypeURI:        func() RR { return new(URI) },
	TypeWKS:        func() RR { return new(WKS) },
	TypeX25:        func() RR { return new(X25) },
}
