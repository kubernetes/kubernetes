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
	TypeSMIMEA     uint16 = 53
	TypeHIP        uint16 = 55
	TypeNINFO      uint16 = 56
	TypeRKEY       uint16 = 57
	TypeTALINK     uint16 = 58
	TypeCDS        uint16 = 59
	TypeCDNSKEY    uint16 = 60
	TypeOPENPGPKEY uint16 = 61
	TypeCSYNC      uint16 = 62
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
	TypeAVC        uint16 = 258

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

	// Message Response Codes, see https://www.iana.org/assignments/dns-parameters/dns-parameters.xhtml
	RcodeSuccess        = 0  // NoError   - No Error                          [DNS]
	RcodeFormatError    = 1  // FormErr   - Format Error                      [DNS]
	RcodeServerFailure  = 2  // ServFail  - Server Failure                    [DNS]
	RcodeNameError      = 3  // NXDomain  - Non-Existent Domain               [DNS]
	RcodeNotImplemented = 4  // NotImp    - Not Implemented                   [DNS]
	RcodeRefused        = 5  // Refused   - Query Refused                     [DNS]
	RcodeYXDomain       = 6  // YXDomain  - Name Exists when it should not    [DNS Update]
	RcodeYXRrset        = 7  // YXRRSet   - RR Set Exists when it should not  [DNS Update]
	RcodeNXRrset        = 8  // NXRRSet   - RR Set that should exist does not [DNS Update]
	RcodeNotAuth        = 9  // NotAuth   - Server Not Authoritative for zone [DNS Update]
	RcodeNotZone        = 10 // NotZone   - Name not contained in zone        [DNS Update/TSIG]
	RcodeBadSig         = 16 // BADSIG    - TSIG Signature Failure            [TSIG]
	RcodeBadVers        = 16 // BADVERS   - Bad OPT Version                   [EDNS0]
	RcodeBadKey         = 17 // BADKEY    - Key not recognized                [TSIG]
	RcodeBadTime        = 18 // BADTIME   - Signature out of time window      [TSIG]
	RcodeBadMode        = 19 // BADMODE   - Bad TKEY Mode                     [TKEY]
	RcodeBadName        = 20 // BADNAME   - Duplicate key name                [TKEY]
	RcodeBadAlg         = 21 // BADALG    - Algorithm not supported           [TKEY]
	RcodeBadTrunc       = 22 // BADTRUNC  - Bad Truncation                    [TSIG]
	RcodeBadCookie      = 23 // BADCOOKIE - Bad/missing Server Cookie         [DNS Cookies]

	// Message Opcodes. There is no 3.
	OpcodeQuery  = 0
	OpcodeIQuery = 1
	OpcodeStatus = 2
	OpcodeNotify = 4
	OpcodeUpdate = 5
)

// Header is the wire format for the DNS packet header.
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
)

// Various constants used in the LOC RR, See RFC 1887.
const (
	LOC_EQUATOR       = 1 << 31 // RFC 1876, Section 2.
	LOC_PRIMEMERIDIAN = 1 << 31 // RFC 1876, Section 2.
	LOC_HOURS         = 60 * 1000
	LOC_DEGREES       = 60 * LOC_HOURS
	LOC_ALTITUDEBASE  = 100000
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

//go:generate go run types_generate.go

// Question holds a DNS question. There can be multiple questions in the
// question section of a message. Usually there is just one.
type Question struct {
	Name   string `dns:"cdomain-name"` // "cdomain-name" specifies encoding (and may be compressed)
	Qtype  uint16
	Qclass uint16
}

func (q *Question) len(off int, compression map[string]struct{}) int {
	l := domainNameLen(q.Name, off, compression, true)
	l += 2 + 2
	return l
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

func (rr *ANY) parse(c *zlexer, origin, file string) *ParseError {
	panic("dns: internal error: parse should never be called on ANY")
}

// NULL RR. See RFC 1035.
type NULL struct {
	Hdr  RR_Header
	Data string `dns:"any"`
}

func (rr *NULL) String() string {
	// There is no presentation format; prefix string with a comment.
	return ";" + rr.Hdr.String() + rr.Data
}

func (rr *NULL) parse(c *zlexer, origin, file string) *ParseError {
	panic("dns: internal error: parse should never be called on NULL")
}

// CNAME RR. See RFC 1034.
type CNAME struct {
	Hdr    RR_Header
	Target string `dns:"cdomain-name"`
}

func (rr *CNAME) String() string { return rr.Hdr.String() + sprintName(rr.Target) }

// HINFO RR. See RFC 1034.
type HINFO struct {
	Hdr RR_Header
	Cpu string
	Os  string
}

func (rr *HINFO) String() string {
	return rr.Hdr.String() + sprintTxt([]string{rr.Cpu, rr.Os})
}

// MB RR. See RFC 1035.
type MB struct {
	Hdr RR_Header
	Mb  string `dns:"cdomain-name"`
}

func (rr *MB) String() string { return rr.Hdr.String() + sprintName(rr.Mb) }

// MG RR. See RFC 1035.
type MG struct {
	Hdr RR_Header
	Mg  string `dns:"cdomain-name"`
}

func (rr *MG) String() string { return rr.Hdr.String() + sprintName(rr.Mg) }

// MINFO RR. See RFC 1035.
type MINFO struct {
	Hdr   RR_Header
	Rmail string `dns:"cdomain-name"`
	Email string `dns:"cdomain-name"`
}

func (rr *MINFO) String() string {
	return rr.Hdr.String() + sprintName(rr.Rmail) + " " + sprintName(rr.Email)
}

// MR RR. See RFC 1035.
type MR struct {
	Hdr RR_Header
	Mr  string `dns:"cdomain-name"`
}

func (rr *MR) String() string {
	return rr.Hdr.String() + sprintName(rr.Mr)
}

// MF RR. See RFC 1035.
type MF struct {
	Hdr RR_Header
	Mf  string `dns:"cdomain-name"`
}

func (rr *MF) String() string {
	return rr.Hdr.String() + sprintName(rr.Mf)
}

// MD RR. See RFC 1035.
type MD struct {
	Hdr RR_Header
	Md  string `dns:"cdomain-name"`
}

func (rr *MD) String() string {
	return rr.Hdr.String() + sprintName(rr.Md)
}

// MX RR. See RFC 1035.
type MX struct {
	Hdr        RR_Header
	Preference uint16
	Mx         string `dns:"cdomain-name"`
}

func (rr *MX) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Mx)
}

// AFSDB RR. See RFC 1183.
type AFSDB struct {
	Hdr      RR_Header
	Subtype  uint16
	Hostname string `dns:"domain-name"`
}

func (rr *AFSDB) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Subtype)) + " " + sprintName(rr.Hostname)
}

// X25 RR. See RFC 1183, Section 3.1.
type X25 struct {
	Hdr         RR_Header
	PSDNAddress string
}

func (rr *X25) String() string {
	return rr.Hdr.String() + rr.PSDNAddress
}

// RT RR. See RFC 1183, Section 3.3.
type RT struct {
	Hdr        RR_Header
	Preference uint16
	Host       string `dns:"domain-name"` // RFC 3597 prohibits compressing records not defined in RFC 1035.
}

func (rr *RT) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Host)
}

// NS RR. See RFC 1035.
type NS struct {
	Hdr RR_Header
	Ns  string `dns:"cdomain-name"`
}

func (rr *NS) String() string {
	return rr.Hdr.String() + sprintName(rr.Ns)
}

// PTR RR. See RFC 1035.
type PTR struct {
	Hdr RR_Header
	Ptr string `dns:"cdomain-name"`
}

func (rr *PTR) String() string {
	return rr.Hdr.String() + sprintName(rr.Ptr)
}

// RP RR. See RFC 1138, Section 2.2.
type RP struct {
	Hdr  RR_Header
	Mbox string `dns:"domain-name"`
	Txt  string `dns:"domain-name"`
}

func (rr *RP) String() string {
	return rr.Hdr.String() + rr.Mbox + " " + sprintTxt([]string{rr.Txt})
}

// SOA RR. See RFC 1035.
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

// TXT RR. See RFC 1035.
type TXT struct {
	Hdr RR_Header
	Txt []string `dns:"txt"`
}

func (rr *TXT) String() string { return rr.Hdr.String() + sprintTxt(rr.Txt) }

func sprintName(s string) string {
	var dst strings.Builder
	dst.Grow(len(s))
	for i := 0; i < len(s); {
		if i+1 < len(s) && s[i] == '\\' && s[i+1] == '.' {
			dst.WriteString(s[i : i+2])
			i += 2
			continue
		}

		b, n := nextByte(s, i)
		switch {
		case n == 0:
			i++ // dangling back slash
		case b == '.':
			dst.WriteByte('.')
		default:
			writeDomainNameByte(&dst, b)
		}
		i += n
	}
	return dst.String()
}

func sprintTxtOctet(s string) string {
	var dst strings.Builder
	dst.Grow(2 + len(s))
	dst.WriteByte('"')
	for i := 0; i < len(s); {
		if i+1 < len(s) && s[i] == '\\' && s[i+1] == '.' {
			dst.WriteString(s[i : i+2])
			i += 2
			continue
		}

		b, n := nextByte(s, i)
		switch {
		case n == 0:
			i++ // dangling back slash
		case b == '.':
			dst.WriteByte('.')
		case b < ' ' || b > '~':
			dst.WriteString(escapeByte(b))
		default:
			dst.WriteByte(b)
		}
		i += n
	}
	dst.WriteByte('"')
	return dst.String()
}

func sprintTxt(txt []string) string {
	var out strings.Builder
	for i, s := range txt {
		out.Grow(3 + len(s))
		if i > 0 {
			out.WriteString(` "`)
		} else {
			out.WriteByte('"')
		}
		for j := 0; j < len(s); {
			b, n := nextByte(s, j)
			if n == 0 {
				break
			}
			writeTXTStringByte(&out, b)
			j += n
		}
		out.WriteByte('"')
	}
	return out.String()
}

func writeDomainNameByte(s *strings.Builder, b byte) {
	switch b {
	case '.', ' ', '\'', '@', ';', '(', ')': // additional chars to escape
		s.WriteByte('\\')
		s.WriteByte(b)
	default:
		writeTXTStringByte(s, b)
	}
}

func writeTXTStringByte(s *strings.Builder, b byte) {
	switch {
	case b == '"' || b == '\\':
		s.WriteByte('\\')
		s.WriteByte(b)
	case b < ' ' || b > '~':
		s.WriteString(escapeByte(b))
	default:
		s.WriteByte(b)
	}
}

const (
	escapedByteSmall = "" +
		`\000\001\002\003\004\005\006\007\008\009` +
		`\010\011\012\013\014\015\016\017\018\019` +
		`\020\021\022\023\024\025\026\027\028\029` +
		`\030\031`
	escapedByteLarge = `\127\128\129` +
		`\130\131\132\133\134\135\136\137\138\139` +
		`\140\141\142\143\144\145\146\147\148\149` +
		`\150\151\152\153\154\155\156\157\158\159` +
		`\160\161\162\163\164\165\166\167\168\169` +
		`\170\171\172\173\174\175\176\177\178\179` +
		`\180\181\182\183\184\185\186\187\188\189` +
		`\190\191\192\193\194\195\196\197\198\199` +
		`\200\201\202\203\204\205\206\207\208\209` +
		`\210\211\212\213\214\215\216\217\218\219` +
		`\220\221\222\223\224\225\226\227\228\229` +
		`\230\231\232\233\234\235\236\237\238\239` +
		`\240\241\242\243\244\245\246\247\248\249` +
		`\250\251\252\253\254\255`
)

// escapeByte returns the \DDD escaping of b which must
// satisfy b < ' ' || b > '~'.
func escapeByte(b byte) string {
	if b < ' ' {
		return escapedByteSmall[b*4 : b*4+4]
	}

	b -= '~' + 1
	// The cast here is needed as b*4 may overflow byte.
	return escapedByteLarge[int(b)*4 : int(b)*4+4]
}

func nextByte(s string, offset int) (byte, int) {
	if offset >= len(s) {
		return 0, 0
	}
	if s[offset] != '\\' {
		// not an escape sequence
		return s[offset], 1
	}
	switch len(s) - offset {
	case 1: // dangling escape
		return 0, 0
	case 2, 3: // too short to be \ddd
	default: // maybe \ddd
		if isDigit(s[offset+1]) && isDigit(s[offset+2]) && isDigit(s[offset+3]) {
			return dddStringToByte(s[offset+1:]), 4
		}
	}
	// not \ddd, just an RFC 1035 "quoted" character
	return s[offset+1], 2
}

// SPF RR. See RFC 4408, Section 3.1.1.
type SPF struct {
	Hdr RR_Header
	Txt []string `dns:"txt"`
}

func (rr *SPF) String() string { return rr.Hdr.String() + sprintTxt(rr.Txt) }

// AVC RR. See https://www.iana.org/assignments/dns-parameters/AVC/avc-completed-template.
type AVC struct {
	Hdr RR_Header
	Txt []string `dns:"txt"`
}

func (rr *AVC) String() string { return rr.Hdr.String() + sprintTxt(rr.Txt) }

// SRV RR. See RFC 2782.
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

// NAPTR RR. See RFC 2915.
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

// CERT RR. See RFC 4398.
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

// DNAME RR. See RFC 2672.
type DNAME struct {
	Hdr    RR_Header
	Target string `dns:"domain-name"`
}

func (rr *DNAME) String() string {
	return rr.Hdr.String() + sprintName(rr.Target)
}

// A RR. See RFC 1035.
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

// AAAA RR. See RFC 3596.
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

// PX RR. See RFC 2163.
type PX struct {
	Hdr        RR_Header
	Preference uint16
	Map822     string `dns:"domain-name"`
	Mapx400    string `dns:"domain-name"`
}

func (rr *PX) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Map822) + " " + sprintName(rr.Mapx400)
}

// GPOS RR. See RFC 1712.
type GPOS struct {
	Hdr       RR_Header
	Longitude string
	Latitude  string
	Altitude  string
}

func (rr *GPOS) String() string {
	return rr.Hdr.String() + rr.Longitude + " " + rr.Latitude + " " + rr.Altitude
}

// LOC RR. See RFC RFC 1876.
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
	s += fmt.Sprintf("%02d %02d %0.3f %s ", h, m, float64(lat)/1000, ns)

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
	s += fmt.Sprintf("%02d %02d %0.3f %s ", h, m, float64(lon)/1000, ew)

	var alt = float64(rr.Altitude) / 100
	alt -= LOC_ALTITUDEBASE
	if rr.Altitude%100 != 0 {
		s += fmt.Sprintf("%.2fm ", alt)
	} else {
		s += fmt.Sprintf("%.0fm ", alt)
	}

	s += cmToM(rr.Size&0xf0>>4, rr.Size&0x0f) + "m "
	s += cmToM(rr.HorizPre&0xf0>>4, rr.HorizPre&0x0f) + "m "
	s += cmToM(rr.VertPre&0xf0>>4, rr.VertPre&0x0f) + "m"

	return s
}

// SIG RR. See RFC 2535. The SIG RR is identical to RRSIG and nowadays only used for SIG(0), See RFC 2931.
type SIG struct {
	RRSIG
}

// RRSIG RR. See RFC 4034 and RFC 3755.
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

// NSEC RR. See RFC 4034 and RFC 3755.
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

func (rr *NSEC) len(off int, compression map[string]struct{}) int {
	l := rr.Hdr.len(off, compression)
	l += domainNameLen(rr.NextDomain, off+l, compression, false)
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

// DLV RR. See RFC 4431.
type DLV struct{ DS }

// CDS RR. See RFC 7344.
type CDS struct{ DS }

// DS RR. See RFC 4034 and RFC 3658.
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

// KX RR. See RFC 2230.
type KX struct {
	Hdr        RR_Header
	Preference uint16
	Exchanger  string `dns:"domain-name"`
}

func (rr *KX) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) +
		" " + sprintName(rr.Exchanger)
}

// TA RR. See http://www.watson.org/~weiler/INI1999-19.pdf.
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

// TALINK RR. See https://www.iana.org/assignments/dns-parameters/TALINK/talink-completed-template.
type TALINK struct {
	Hdr          RR_Header
	PreviousName string `dns:"domain-name"`
	NextName     string `dns:"domain-name"`
}

func (rr *TALINK) String() string {
	return rr.Hdr.String() +
		sprintName(rr.PreviousName) + " " + sprintName(rr.NextName)
}

// SSHFP RR. See RFC RFC 4255.
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

// KEY RR. See RFC RFC 2535.
type KEY struct {
	DNSKEY
}

// CDNSKEY RR. See RFC 7344.
type CDNSKEY struct {
	DNSKEY
}

// DNSKEY RR. See RFC 4034 and RFC 3755.
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

// RKEY RR. See https://www.iana.org/assignments/dns-parameters/RKEY/rkey-completed-template.
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

// NSAPPTR RR. See RFC 1348.
type NSAPPTR struct {
	Hdr RR_Header
	Ptr string `dns:"domain-name"`
}

func (rr *NSAPPTR) String() string { return rr.Hdr.String() + sprintName(rr.Ptr) }

// NSEC3 RR. See RFC 5155.
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

func (rr *NSEC3) len(off int, compression map[string]struct{}) int {
	l := rr.Hdr.len(off, compression)
	l += 6 + len(rr.Salt)/2 + 1 + len(rr.NextDomain) + 1
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

// NSEC3PARAM RR. See RFC 5155.
type NSEC3PARAM struct {
	Hdr        RR_Header
	Hash       uint8
	Flags      uint8
	Iterations uint16
	SaltLength uint8
	Salt       string `dns:"size-hex:SaltLength"`
}

func (rr *NSEC3PARAM) String() string {
	s := rr.Hdr.String()
	s += strconv.Itoa(int(rr.Hash)) +
		" " + strconv.Itoa(int(rr.Flags)) +
		" " + strconv.Itoa(int(rr.Iterations)) +
		" " + saltToString(rr.Salt)
	return s
}

// TKEY RR. See RFC 2930.
type TKEY struct {
	Hdr        RR_Header
	Algorithm  string `dns:"domain-name"`
	Inception  uint32
	Expiration uint32
	Mode       uint16
	Error      uint16
	KeySize    uint16
	Key        string `dns:"size-hex:KeySize"`
	OtherLen   uint16
	OtherData  string `dns:"size-hex:OtherLen"`
}

// TKEY has no official presentation format, but this will suffice.
func (rr *TKEY) String() string {
	s := ";" + rr.Hdr.String() +
		" " + rr.Algorithm +
		" " + TimeToString(rr.Inception) +
		" " + TimeToString(rr.Expiration) +
		" " + strconv.Itoa(int(rr.Mode)) +
		" " + strconv.Itoa(int(rr.Error)) +
		" " + strconv.Itoa(int(rr.KeySize)) +
		" " + rr.Key +
		" " + strconv.Itoa(int(rr.OtherLen)) +
		" " + rr.OtherData
	return s
}

// RFC3597 represents an unknown/generic RR. See RFC 3597.
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

// URI RR. See RFC 7553.
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

// DHCID RR. See RFC 4701.
type DHCID struct {
	Hdr    RR_Header
	Digest string `dns:"base64"`
}

func (rr *DHCID) String() string { return rr.Hdr.String() + rr.Digest }

// TLSA RR. See RFC 6698.
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

// SMIMEA RR. See RFC 8162.
type SMIMEA struct {
	Hdr          RR_Header
	Usage        uint8
	Selector     uint8
	MatchingType uint8
	Certificate  string `dns:"hex"`
}

func (rr *SMIMEA) String() string {
	s := rr.Hdr.String() +
		strconv.Itoa(int(rr.Usage)) +
		" " + strconv.Itoa(int(rr.Selector)) +
		" " + strconv.Itoa(int(rr.MatchingType))

	// Every Nth char needs a space on this output. If we output
	// this as one giant line, we can't read it can in because in some cases
	// the cert length overflows scan.maxTok (2048).
	sx := splitN(rr.Certificate, 1024) // conservative value here
	s += " " + strings.Join(sx, " ")
	return s
}

// HIP RR. See RFC 8005.
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

// NINFO RR. See https://www.iana.org/assignments/dns-parameters/NINFO/ninfo-completed-template.
type NINFO struct {
	Hdr    RR_Header
	ZSData []string `dns:"txt"`
}

func (rr *NINFO) String() string { return rr.Hdr.String() + sprintTxt(rr.ZSData) }

// NID RR. See RFC RFC 6742.
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

// L32 RR, See RFC 6742.
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

// L64 RR, See RFC 6742.
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

// LP RR. See RFC 6742.
type LP struct {
	Hdr        RR_Header
	Preference uint16
	Fqdn       string `dns:"domain-name"`
}

func (rr *LP) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Preference)) + " " + sprintName(rr.Fqdn)
}

// EUI48 RR. See RFC 7043.
type EUI48 struct {
	Hdr     RR_Header
	Address uint64 `dns:"uint48"`
}

func (rr *EUI48) String() string { return rr.Hdr.String() + euiToString(rr.Address, 48) }

// EUI64 RR. See RFC 7043.
type EUI64 struct {
	Hdr     RR_Header
	Address uint64
}

func (rr *EUI64) String() string { return rr.Hdr.String() + euiToString(rr.Address, 64) }

// CAA RR. See RFC 6844.
type CAA struct {
	Hdr   RR_Header
	Flag  uint8
	Tag   string
	Value string `dns:"octet"`
}

func (rr *CAA) String() string {
	return rr.Hdr.String() + strconv.Itoa(int(rr.Flag)) + " " + rr.Tag + " " + sprintTxtOctet(rr.Value)
}

// UID RR. Deprecated, IANA-Reserved.
type UID struct {
	Hdr RR_Header
	Uid uint32
}

func (rr *UID) String() string { return rr.Hdr.String() + strconv.FormatInt(int64(rr.Uid), 10) }

// GID RR. Deprecated, IANA-Reserved.
type GID struct {
	Hdr RR_Header
	Gid uint32
}

func (rr *GID) String() string { return rr.Hdr.String() + strconv.FormatInt(int64(rr.Gid), 10) }

// UINFO RR. Deprecated, IANA-Reserved.
type UINFO struct {
	Hdr   RR_Header
	Uinfo string
}

func (rr *UINFO) String() string { return rr.Hdr.String() + sprintTxt([]string{rr.Uinfo}) }

// EID RR. See http://ana-3.lcs.mit.edu/~jnc/nimrod/dns.txt.
type EID struct {
	Hdr      RR_Header
	Endpoint string `dns:"hex"`
}

func (rr *EID) String() string { return rr.Hdr.String() + strings.ToUpper(rr.Endpoint) }

// NIMLOC RR. See http://ana-3.lcs.mit.edu/~jnc/nimrod/dns.txt.
type NIMLOC struct {
	Hdr     RR_Header
	Locator string `dns:"hex"`
}

func (rr *NIMLOC) String() string { return rr.Hdr.String() + strings.ToUpper(rr.Locator) }

// OPENPGPKEY RR. See RFC 7929.
type OPENPGPKEY struct {
	Hdr       RR_Header
	PublicKey string `dns:"base64"`
}

func (rr *OPENPGPKEY) String() string { return rr.Hdr.String() + rr.PublicKey }

// CSYNC RR. See RFC 7477.
type CSYNC struct {
	Hdr        RR_Header
	Serial     uint32
	Flags      uint16
	TypeBitMap []uint16 `dns:"nsec"`
}

func (rr *CSYNC) String() string {
	s := rr.Hdr.String() + strconv.FormatInt(int64(rr.Serial), 10) + " " + strconv.Itoa(int(rr.Flags))

	for i := 0; i < len(rr.TypeBitMap); i++ {
		s += " " + Type(rr.TypeBitMap[i]).String()
	}
	return s
}

func (rr *CSYNC) len(off int, compression map[string]struct{}) int {
	l := rr.Hdr.len(off, compression)
	l += 4 + 2
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

// TimeToString translates the RRSIG's incep. and expir. times to the
// string representation used when printing the record.
// It takes serial arithmetic (RFC 1982) into account.
func TimeToString(t uint32) string {
	mod := (int64(t)-time.Now().Unix())/year68 - 1
	if mod < 0 {
		mod = 0
	}
	ti := time.Unix(int64(t)-mod*year68, 0).UTC()
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
	mod := t.Unix()/year68 - 1
	if mod < 0 {
		mod = 0
	}
	return uint32(t.Unix() - mod*year68), nil
}

// saltToString converts a NSECX salt to uppercase and returns "-" when it is empty.
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

// SplitN splits a string into N sized string chunks.
// This might become an exported function once.
func splitN(s string, n int) []string {
	if len(s) < n {
		return []string{s}
	}
	sx := []string{}
	p, i := 0, n
	for {
		if i <= len(s) {
			sx = append(sx, s[p:i])
		} else {
			sx = append(sx, s[p:])
			break

		}
		p, i = p+n, i+n
	}

	return sx
}
