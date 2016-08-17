package dns

import (
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
	TypeURI        uint16 = 256
	TypeCAA        uint16 = 257

	TypeTKEY uint16 = 249
	TypeTSIG uint16 = 250

	// valid Question.Qtype only
	TypeIXFR  uint16 = 251
	TypeAXFR  uint16 = 252
	TypeMAILB uint16 = 253
	TypeMAILA uint16 = 254
	TypeANY   uint16 = 255

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

	// Message Response Codes.
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
	RcodeBadCookie      = 23 // DNS Cookies

	// Message Opcodes. There is no 3.
	OpcodeQuery  = 0
	OpcodeIQuery = 1
	OpcodeStatus = 2
	OpcodeNotify = 4
	OpcodeUpdate = 5
)

// Headers is the wire format for the DNS packet header.
type Header struct {
	Id                                 uint16
	Bits                               uint16
	Qdcount, Ancount, Nscount, Arcount uint16
}

const (
	headerSize = 12

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

// Different Certificate Types, see RFC 4398, Section 2.1
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

// CertTypeToString converts the Cert Type to its string representation.
// See RFC 4398 and RFC 6944.
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

// StringToCertType is the reverseof CertTypeToString.
var StringToCertType = reverseInt16(CertTypeToString)

//go:generate go run types_generate.go

// Question holds a DNS question. There can be multiple questions in the
// question section of a message. Usually there is just one.
type Question struct {
	Name   string `dns:"cdomain-name"` // "cdomain-name" specifies encoding (and may be compressed)
	Qtype  uint16
	Qclass uint16
}

func (q *Question) len() int {
	return len(q.Name) + 1 + 2 + 2
}

func (q *Question) String() (s string) {
	// prefix with ; (as in dig)
	s = ";" + sprintName(q.Name) + "\t"
	s += Class(q.Qclass).String() + "\t"
	s += " " + Type(q.Qtype).String()
	return s
}

// ANY is a wildcard record. See RFC 1035, Section 3.2.3. ANY
// is named "*" there.
type ANY struct {
	Hdr RR_Header
	// Does not have any rdata
}

func (rr *ANY) String() string { return rr.Hdr.String() }

type CNAME struct {
	Hdr    RR_Header
	Target string `dns:"cdomain-name"`
}

func (rr *CNAME) String() string { return rr.Hdr.String() + sprintName(rr.Target) }

type HINFO struct {
	Hdr RR_Header
	Cpu string
	Os  string
}

func (rr *HINFO) String() string {
	return rr.Hdr.String() + sprintTxt([]string{rr.Cpu, rr.Os})
}

type MB struct {
	Hdr RR_Header
	Mb  string `dns:"cdomain-name"`
}

func (rr *MB) String() string { return rr.Hdr.String() + sprintName(rr.Mb) }

type MG struct {
	Hdr RR_Header
	Mg  string `dns:"cdomain-name"`
}

func (rr *MG) String() string { return rr.Hdr.String() + sprintName(rr.Mg) }

type MINFO struct {
	Hdr   RR_Header
	Rmail string `dns:"cdomain-name"`
	Email string `dns:"cdomain-name"`
}

func (rr *MINFO) String() string {
	return rr.Hdr.String() + sprintName(rr.Rmail) + " " + sprintName(rr.Email)
}

type MR struct {
	Hdr RR_Header
	Mr  string `dns:"cdomain-name"`
}

func (rr *MR) String() string {
	return rr.Hdr.String() + sprintName(rr.Mr)
}

type MF struct {
	Hdr RR_Header
	Mf  string `dns:"cdomain-name"`
}

func (rr *MF) String() string {
	return rr.Hdr.String() + sprintName(rr.Mf)
}

type MD struct {
	Hdr RR_Header
	Md  string `dns:"cdomain-name"`
}

func (rr *MD) String() string {
	return rr.Hdr.String() + sprintName(rr.Md)
}

type MX struct {
	Hdr        RR_Header
	Preference uint16
	Mx         string `dns:"cdomain-name"`
}

func (rr *MX) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Mx)
}

type AFSDB struct {
	Hdr      RR_Header
	Subtype  uint16
	Hostname string `dns:"cdomain-name"`
}

func (rr *AFSDB) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Subtype)) + " " + sprintName(rr.Hostname)
}

type X25 struct {
	Hdr         RR_Header
	PSDNAddress string
}

func (rr *X25) String() string {
	return rr.Hdr.String() + rr.PSDNAddress
}

type RT struct {
	Hdr        RR_Header
	Preference uint16
	Host       string `dns:"cdomain-name"`
}

func (rr *RT) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Host)
}

type NS struct {
	Hdr RR_Header
	Ns  string `dns:"cdomain-name"`
}

func (rr *NS) String() string {
	return rr.Hdr.String() + sprintName(rr.Ns)
}

type PTR struct {
	Hdr RR_Header
	Ptr string `dns:"cdomain-name"`
}

func (rr *PTR) String() string {
	return rr.Hdr.String() + sprintName(rr.Ptr)
}

type RP struct {
	Hdr  RR_Header
	Mbox string `dns:"domain-name"`
	Txt  string `dns:"domain-name"`
}

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

func (rr *SOA) String() string {
	return rr.Hdr.String() + sprintName(rr.Ns) + " " + sprintName(rr.Mbox) +
		" " + strconv.FormatInt(int64(rr.Serial), 10) +
		" " + strconv.FormatInt(int64(rr.Refresh), 10) +
		" " + strconv.FormatInt(int64(rr.Retry), 10) +
		" " + strconv.FormatInt(int64(rr.Expire), 10) +
		" " + strconv.FormatInt(int64(rr.Minttl), 10)
}

type TXT struct {
	Hdr RR_Header
	Txt []string `dns:"txt"`
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

func sprintTxtOctet(s string) string {
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

type SPF struct {
	Hdr RR_Header
	Txt []string `dns:"txt"`
}

func (rr *SPF) String() string { return rr.Hdr.String() + sprintTxt(rr.Txt) }

type SRV struct {
	Hdr      RR_Header
	Priority uint16
	Weight   uint16
	Port     uint16
	Target   string `dns:"domain-name"`
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

func (rr *NAPTR) String() string {
	return rr.Hdr.String() +
		strconv.Itoa(int(rr.Order)) + " " +
		strconv.Itoa(int(rr.Preference)) + " " +
		"\"" + rr.Flags + "\" " +
		"\"" + rr.Service + "\" " +
		"\"" + rr.Regexp + "\" " +
		rr.Replacement
}

// The CERT resource record, see RFC 4398.
type CERT struct {
	Hdr         RR_Header
	Type        uint16
	KeyTag      uint16
	Algorithm   uint8
	Certificate string `dns:"base64"`
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

// The DNAME resource record, see RFC 2672.
type DNAME struct {
	Hdr    RR_Header
	Target string `dns:"domain-name"`
}

func (rr *DNAME) String() string {
	return rr.Hdr.String() + sprintName(rr.Target)
}

type A struct {
	Hdr RR_Header
	A   net.IP `dns:"a"`
}

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

func (rr *PX) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Map822) + " " + sprintName(rr.Mapx400)
}

type GPOS struct {
	Hdr       RR_Header
	Longitude string
	Latitude  string
	Altitude  string
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

type NSEC struct {
	Hdr        RR_Header
	NextDomain string   `dns:"domain-name"`
	TypeBitMap []uint16 `dns:"nsec"`
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

func (rr *SSHFP) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Algorithm)) +
		" " + strconv.Itoa(int(rr.Type)) +
		" " + strings.ToUpper(rr.FingerPrint)
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

func (rr *RKEY) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Flags)) +
		" " + strconv.Itoa(int(rr.Protocol)) +
		" " + strconv.Itoa(int(rr.Algorithm)) +
		" " + rr.PublicKey
}

type NSAPPTR struct {
	Hdr RR_Header
	Ptr string `dns:"domain-name"`
}

func (rr *NSAPPTR) String() string { return rr.Hdr.String() + sprintName(rr.Ptr) }

type NSEC3 struct {
	Hdr        RR_Header
	Hash       uint8
	Flags      uint8
	Iterations uint16
	SaltLength uint8
	Salt       string `dns:"size-hex:SaltLength"`
	HashLength uint8
	NextDomain string   `dns:"size-base32:HashLength"`
	TypeBitMap []uint16 `dns:"nsec"`
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

func (rr *TKEY) String() string {
	// It has no presentation format
	return ""
}

// RFC3597 represents an unknown/generic RR.
type RFC3597 struct {
	Hdr   RR_Header
	Rdata string `dns:"hex"`
}

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
	Target   string `dns:"octet"`
}

func (rr *URI) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Priority)) +
		" " + strconv.Itoa(int(rr.Weight)) + " " + sprintTxtOctet(rr.Target)
}

type DHCID struct {
	Hdr    RR_Header
	Digest string `dns:"base64"`
}

func (rr *DHCID) String() string { return rr.Hdr.String() + rr.Digest }

type TLSA struct {
	Hdr          RR_Header
	Usage        uint8
	Selector     uint8
	MatchingType uint8
	Certificate  string `dns:"hex"`
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
	Hit                string   `dns:"size-hex:HitLength"`
	PublicKey          string   `dns:"size-base64:PublicKeyLength"`
	RendezvousServers  []string `dns:"domain-name"`
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

type NINFO struct {
	Hdr    RR_Header
	ZSData []string `dns:"txt"`
}

func (rr *NINFO) String() string { return rr.Hdr.String() + sprintTxt(rr.ZSData) }

type NID struct {
	Hdr        RR_Header
	Preference uint16
	NodeID     uint64
}

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

func (rr *LP) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Fqdn)
}

type EUI48 struct {
	Hdr     RR_Header
	Address uint64 `dns:"uint48"`
}

func (rr *EUI48) String() string { return rr.Hdr.String() + euiToString(rr.Address, 48) }

type EUI64 struct {
	Hdr     RR_Header
	Address uint64
}

func (rr *EUI64) String() string { return rr.Hdr.String() + euiToString(rr.Address, 64) }

type CAA struct {
	Hdr   RR_Header
	Flag  uint8
	Tag   string
	Value string `dns:"octet"`
}

func (rr *CAA) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Flag)) + " " + rr.Tag + " " + sprintTxtOctet(rr.Value)
}

type UID struct {
	Hdr RR_Header
	Uid uint32
}

func (rr *UID) String() string { return rr.Hdr.String() + strconv.FormatInt(int64(rr.Uid), 10) }

type GID struct {
	Hdr RR_Header
	Gid uint32
}

func (rr *GID) String() string { return rr.Hdr.String() + strconv.FormatInt(int64(rr.Gid), 10) }

type UINFO struct {
	Hdr   RR_Header
	Uinfo string
}

func (rr *UINFO) String() string { return rr.Hdr.String() + sprintTxt([]string{rr.Uinfo}) }

type EID struct {
	Hdr      RR_Header
	Endpoint string `dns:"hex"`
}

func (rr *EID) String() string { return rr.Hdr.String() + strings.ToUpper(rr.Endpoint) }

type NIMLOC struct {
	Hdr     RR_Header
	Locator string `dns:"hex"`
}

func (rr *NIMLOC) String() string { return rr.Hdr.String() + strings.ToUpper(rr.Locator) }

type OPENPGPKEY struct {
	Hdr       RR_Header
	PublicKey string `dns:"base64"`
}

func (rr *OPENPGPKEY) String() string { return rr.Hdr.String() + rr.PublicKey }

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
	t, err := time.Parse("20060102150405", s)
	if err != nil {
		return 0, err
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
