// DNS packet assembly, see RFC 1035. Converting from - Unpack() -
// and to - Pack() - wire format.
// All the packers and unpackers take a (msg []byte, off int)
// and return (off1 int, ok bool).  If they return ok==false, they
// also return off1==len(msg), so that the next unpacker will
// also fail.  This lets us avoid checks of ok until the end of a
// packing sequence.

package dns

import (
	"encoding/base32"
	"encoding/base64"
	"encoding/hex"
	"math/big"
	"math/rand"
	"net"
	"reflect"
	"strconv"
	"time"
)

const maxCompressionOffset = 2 << 13 // We have 14 bits for the compression pointer

var (
	// ErrAlg indicates an error with the (DNSSEC) algorithm.
	ErrAlg error = &Error{err: "bad algorithm"}
	// ErrAuth indicates an error in the TSIG authentication.
	ErrAuth error = &Error{err: "bad authentication"}
	// ErrBuf indicates that the buffer used it too small for the message.
	ErrBuf error = &Error{err: "buffer size too small"}
	// ErrConn indicates that a connection has both a TCP and UDP socket.
	ErrConn error = &Error{err: "conn holds both UDP and TCP connection"}
	// ErrConnEmpty indicates a connection is being uses before it is initialized.
	ErrConnEmpty error = &Error{err: "conn has no connection"}
	// ErrExtendedRcode ...
	ErrExtendedRcode error = &Error{err: "bad extended rcode"}
	// ErrFqdn indicates that a domain name does not have a closing dot.
	ErrFqdn error = &Error{err: "domain must be fully qualified"}
	// ErrId indicates there is a mismatch with the message's ID.
	ErrId        error = &Error{err: "id mismatch"}
	ErrKeyAlg    error = &Error{err: "bad key algorithm"}
	ErrKey       error = &Error{err: "bad key"}
	ErrKeySize   error = &Error{err: "bad key size"}
	ErrNoSig     error = &Error{err: "no signature found"}
	ErrPrivKey   error = &Error{err: "bad private key"}
	ErrRcode     error = &Error{err: "bad rcode"}
	ErrRdata     error = &Error{err: "bad rdata"}
	ErrRRset     error = &Error{err: "bad rrset"}
	ErrSecret    error = &Error{err: "no secrets defined"}
	ErrShortRead error = &Error{err: "short read"}
	// ErrSig indicates that a signature can not be cryptographically validated.
	ErrSig error = &Error{err: "bad signature"}
	// ErrSigGen indicates a faulure to generate a signature.
	ErrSigGen error = &Error{err: "bad signature generation"}
	// ErrSOA indicates that no SOA RR was seen when doing zone transfers.
	ErrSoa error = &Error{err: "no SOA"}
	// ErrTime indicates a timing error in TSIG authentication.
	ErrTime error = &Error{err: "bad time"}
)

// Id, by default, returns a 16 bits random number to be used as a
// message id. The random provided should be good enough. This being a
// variable the function can be reassigned to a custom function.
// For instance, to make it return a static value:
//
//	dns.Id = func() uint16 { return 3 }
var Id func() uint16 = id

// MsgHdr is a a manually-unpacked version of (id, bits).
type MsgHdr struct {
	Id                 uint16
	Response           bool
	Opcode             int
	Authoritative      bool
	Truncated          bool
	RecursionDesired   bool
	RecursionAvailable bool
	Zero               bool
	AuthenticatedData  bool
	CheckingDisabled   bool
	Rcode              int
}

// Msg contains the layout of a DNS message.
type Msg struct {
	MsgHdr
	Compress bool       `json:"-"` // If true, the message will be compressed when converted to wire format. This not part of the official DNS packet format.
	Question []Question // Holds the RR(s) of the question section.
	Answer   []RR       // Holds the RR(s) of the answer section.
	Ns       []RR       // Holds the RR(s) of the authority section.
	Extra    []RR       // Holds the RR(s) of the additional section.
}

// TypeToString is a map of strings for each RR wire type.
var TypeToString = map[uint16]string{
	TypeA:          "A",
	TypeAAAA:       "AAAA",
	TypeAFSDB:      "AFSDB",
	TypeANY:        "ANY", // Meta RR
	TypeATMA:       "ATMA",
	TypeAXFR:       "AXFR", // Meta RR
	TypeCAA:        "CAA",
	TypeCDNSKEY:    "CDNSKEY",
	TypeCDS:        "CDS",
	TypeCERT:       "CERT",
	TypeCNAME:      "CNAME",
	TypeDHCID:      "DHCID",
	TypeDLV:        "DLV",
	TypeDNAME:      "DNAME",
	TypeDNSKEY:     "DNSKEY",
	TypeDS:         "DS",
	TypeEID:        "EID",
	TypeEUI48:      "EUI48",
	TypeEUI64:      "EUI64",
	TypeGID:        "GID",
	TypeGPOS:       "GPOS",
	TypeHINFO:      "HINFO",
	TypeHIP:        "HIP",
	TypeIPSECKEY:   "IPSECKEY",
	TypeISDN:       "ISDN",
	TypeIXFR:       "IXFR", // Meta RR
	TypeKEY:        "KEY",
	TypeKX:         "KX",
	TypeL32:        "L32",
	TypeL64:        "L64",
	TypeLOC:        "LOC",
	TypeLP:         "LP",
	TypeMB:         "MB",
	TypeMD:         "MD",
	TypeMF:         "MF",
	TypeMG:         "MG",
	TypeMINFO:      "MINFO",
	TypeMR:         "MR",
	TypeMX:         "MX",
	TypeNAPTR:      "NAPTR",
	TypeNID:        "NID",
	TypeNINFO:      "NINFO",
	TypeNIMLOC:     "NIMLOC",
	TypeNS:         "NS",
	TypeNSAP:       "NSAP",
	TypeNSAPPTR:    "NSAP-PTR",
	TypeNSEC3:      "NSEC3",
	TypeNSEC3PARAM: "NSEC3PARAM",
	TypeNSEC:       "NSEC",
	TypeNULL:       "NULL",
	TypeOPT:        "OPT",
	TypeOPENPGPKEY: "OPENPGPKEY",
	TypePTR:        "PTR",
	TypeRKEY:       "RKEY",
	TypeRP:         "RP",
	TypeRRSIG:      "RRSIG",
	TypeRT:         "RT",
	TypeSIG:        "SIG",
	TypeSOA:        "SOA",
	TypeSPF:        "SPF",
	TypeSRV:        "SRV",
	TypeSSHFP:      "SSHFP",
	TypeTA:         "TA",
	TypeTALINK:     "TALINK",
	TypeTKEY:       "TKEY", // Meta RR
	TypeTLSA:       "TLSA",
	TypeTSIG:       "TSIG", // Meta RR
	TypeTXT:        "TXT",
	TypePX:         "PX",
	TypeUID:        "UID",
	TypeUINFO:      "UINFO",
	TypeUNSPEC:     "UNSPEC",
	TypeURI:        "URI",
	TypeWKS:        "WKS",
	TypeX25:        "X25",
}

// StringToType is the reverse of TypeToString, needed for string parsing.
var StringToType = reverseInt16(TypeToString)

// StringToClass is the reverse of ClassToString, needed for string parsing.
var StringToClass = reverseInt16(ClassToString)

// Map of opcodes strings.
var StringToOpcode = reverseInt(OpcodeToString)

// Map of rcodes strings.
var StringToRcode = reverseInt(RcodeToString)

// ClassToString is a maps Classes to strings for each CLASS wire type.
var ClassToString = map[uint16]string{
	ClassINET:   "IN",
	ClassCSNET:  "CS",
	ClassCHAOS:  "CH",
	ClassHESIOD: "HS",
	ClassNONE:   "NONE",
	ClassANY:    "ANY",
}

// OpcodeToString maps Opcodes to strings.
var OpcodeToString = map[int]string{
	OpcodeQuery:  "QUERY",
	OpcodeIQuery: "IQUERY",
	OpcodeStatus: "STATUS",
	OpcodeNotify: "NOTIFY",
	OpcodeUpdate: "UPDATE",
}

// RcodeToString maps Rcodes to strings.
var RcodeToString = map[int]string{
	RcodeSuccess:        "NOERROR",
	RcodeFormatError:    "FORMERR",
	RcodeServerFailure:  "SERVFAIL",
	RcodeNameError:      "NXDOMAIN",
	RcodeNotImplemented: "NOTIMPL",
	RcodeRefused:        "REFUSED",
	RcodeYXDomain:       "YXDOMAIN", // From RFC 2136
	RcodeYXRrset:        "YXRRSET",
	RcodeNXRrset:        "NXRRSET",
	RcodeNotAuth:        "NOTAUTH",
	RcodeNotZone:        "NOTZONE",
	RcodeBadSig:         "BADSIG", // Also known as RcodeBadVers, see RFC 6891
	//	RcodeBadVers:        "BADVERS",
	RcodeBadKey:   "BADKEY",
	RcodeBadTime:  "BADTIME",
	RcodeBadMode:  "BADMODE",
	RcodeBadName:  "BADNAME",
	RcodeBadAlg:   "BADALG",
	RcodeBadTrunc: "BADTRUNC",
}

// Rather than write the usual handful of routines to pack and
// unpack every message that can appear on the wire, we use
// reflection to write a generic pack/unpack for structs and then
// use it. Thus, if in the future we need to define new message
// structs, no new pack/unpack/printing code needs to be written.

// Domain names are a sequence of counted strings
// split at the dots. They end with a zero-length string.

// PackDomainName packs a domain name s into msg[off:].
// If compression is wanted compress must be true and the compression
// map needs to hold a mapping between domain names and offsets
// pointing into msg[].
func PackDomainName(s string, msg []byte, off int, compression map[string]int, compress bool) (off1 int, err error) {
	off1, _, err = packDomainName(s, msg, off, compression, compress)
	return
}

func packDomainName(s string, msg []byte, off int, compression map[string]int, compress bool) (off1 int, labels int, err error) {
	// special case if msg == nil
	lenmsg := 256
	if msg != nil {
		lenmsg = len(msg)
	}
	ls := len(s)
	if ls == 0 { // Ok, for instance when dealing with update RR without any rdata.
		return off, 0, nil
	}
	// If not fully qualified, error out, but only if msg == nil #ugly
	switch {
	case msg == nil:
		if s[ls-1] != '.' {
			s += "."
			ls++
		}
	case msg != nil:
		if s[ls-1] != '.' {
			return lenmsg, 0, ErrFqdn
		}
	}
	// Each dot ends a segment of the name.
	// We trade each dot byte for a length byte.
	// Except for escaped dots (\.), which are normal dots.
	// There is also a trailing zero.

	// Compression
	nameoffset := -1
	pointer := -1
	// Emit sequence of counted strings, chopping at dots.
	begin := 0
	bs := []byte(s)
	roBs, bsFresh, escapedDot := s, true, false
	for i := 0; i < ls; i++ {
		if bs[i] == '\\' {
			for j := i; j < ls-1; j++ {
				bs[j] = bs[j+1]
			}
			ls--
			if off+1 > lenmsg {
				return lenmsg, labels, ErrBuf
			}
			// check for \DDD
			if i+2 < ls && isDigit(bs[i]) && isDigit(bs[i+1]) && isDigit(bs[i+2]) {
				bs[i] = dddToByte(bs[i:])
				for j := i + 1; j < ls-2; j++ {
					bs[j] = bs[j+2]
				}
				ls -= 2
			} else if bs[i] == 't' {
				bs[i] = '\t'
			} else if bs[i] == 'r' {
				bs[i] = '\r'
			} else if bs[i] == 'n' {
				bs[i] = '\n'
			}
			escapedDot = bs[i] == '.'
			bsFresh = false
			continue
		}

		if bs[i] == '.' {
			if i > 0 && bs[i-1] == '.' && !escapedDot {
				// two dots back to back is not legal
				return lenmsg, labels, ErrRdata
			}
			if i-begin >= 1<<6 { // top two bits of length must be clear
				return lenmsg, labels, ErrRdata
			}
			// off can already (we're in a loop) be bigger than len(msg)
			// this happens when a name isn't fully qualified
			if off+1 > lenmsg {
				return lenmsg, labels, ErrBuf
			}
			if msg != nil {
				msg[off] = byte(i - begin)
			}
			offset := off
			off++
			for j := begin; j < i; j++ {
				if off+1 > lenmsg {
					return lenmsg, labels, ErrBuf
				}
				if msg != nil {
					msg[off] = bs[j]
				}
				off++
			}
			if compress && !bsFresh {
				roBs = string(bs)
				bsFresh = true
			}
			// Dont try to compress '.'
			if compress && roBs[begin:] != "." {
				if p, ok := compression[roBs[begin:]]; !ok {
					// Only offsets smaller than this can be used.
					if offset < maxCompressionOffset {
						compression[roBs[begin:]] = offset
					}
				} else {
					// The first hit is the longest matching dname
					// keep the pointer offset we get back and store
					// the offset of the current name, because that's
					// where we need to insert the pointer later

					// If compress is true, we're allowed to compress this dname
					if pointer == -1 && compress {
						pointer = p         // Where to point to
						nameoffset = offset // Where to point from
						break
					}
				}
			}
			labels++
			begin = i + 1
		}
		escapedDot = false
	}
	// Root label is special
	if len(bs) == 1 && bs[0] == '.' {
		return off, labels, nil
	}
	// If we did compression and we find something add the pointer here
	if pointer != -1 {
		// We have two bytes (14 bits) to put the pointer in
		// if msg == nil, we will never do compression
		msg[nameoffset], msg[nameoffset+1] = packUint16(uint16(pointer ^ 0xC000))
		off = nameoffset + 1
		goto End
	}
	if msg != nil {
		msg[off] = 0
	}
End:
	off++
	return off, labels, nil
}

// Unpack a domain name.
// In addition to the simple sequences of counted strings above,
// domain names are allowed to refer to strings elsewhere in the
// packet, to avoid repeating common suffixes when returning
// many entries in a single domain.  The pointers are marked
// by a length byte with the top two bits set.  Ignoring those
// two bits, that byte and the next give a 14 bit offset from msg[0]
// where we should pick up the trail.
// Note that if we jump elsewhere in the packet,
// we return off1 == the offset after the first pointer we found,
// which is where the next record will start.
// In theory, the pointers are only allowed to jump backward.
// We let them jump anywhere and stop jumping after a while.

// UnpackDomainName unpacks a domain name into a string.
func UnpackDomainName(msg []byte, off int) (string, int, error) {
	s := make([]byte, 0, 64)
	off1 := 0
	lenmsg := len(msg)
	ptr := 0 // number of pointers followed
Loop:
	for {
		if off >= lenmsg {
			return "", lenmsg, ErrBuf
		}
		c := int(msg[off])
		off++
		switch c & 0xC0 {
		case 0x00:
			if c == 0x00 {
				// end of name
				if len(s) == 0 {
					return ".", off, nil
				}
				break Loop
			}
			// literal string
			if off+c > lenmsg {
				return "", lenmsg, ErrBuf
			}
			for j := off; j < off+c; j++ {
				switch b := msg[j]; b {
				case '.', '(', ')', ';', ' ', '@':
					fallthrough
				case '"', '\\':
					s = append(s, '\\', b)
				case '\t':
					s = append(s, '\\', 't')
				case '\r':
					s = append(s, '\\', 'r')
				default:
					if b < 32 || b >= 127 { // unprintable use \DDD
						var buf [3]byte
						bufs := strconv.AppendInt(buf[:0], int64(b), 10)
						s = append(s, '\\')
						for i := 0; i < 3-len(bufs); i++ {
							s = append(s, '0')
						}
						for _, r := range bufs {
							s = append(s, r)
						}
					} else {
						s = append(s, b)
					}
				}
			}
			s = append(s, '.')
			off += c
		case 0xC0:
			// pointer to somewhere else in msg.
			// remember location after first ptr,
			// since that's how many bytes we consumed.
			// also, don't follow too many pointers --
			// maybe there's a loop.
			if off >= lenmsg {
				return "", lenmsg, ErrBuf
			}
			c1 := msg[off]
			off++
			if ptr == 0 {
				off1 = off
			}
			if ptr++; ptr > 10 {
				return "", lenmsg, &Error{err: "too many compression pointers"}
			}
			off = (c^0xC0)<<8 | int(c1)
		default:
			// 0x80 and 0x40 are reserved
			return "", lenmsg, ErrRdata
		}
	}
	if ptr == 0 {
		off1 = off
	}
	return string(s), off1, nil
}

func packTxt(txt []string, msg []byte, offset int, tmp []byte) (int, error) {
	var err error
	if len(txt) == 0 {
		if offset >= len(msg) {
			return offset, ErrBuf
		}
		msg[offset] = 0
		return offset, nil
	}
	for i := range txt {
		if len(txt[i]) > len(tmp) {
			return offset, ErrBuf
		}
		offset, err = packTxtString(txt[i], msg, offset, tmp)
		if err != nil {
			return offset, err
		}
	}
	return offset, err
}

func packTxtString(s string, msg []byte, offset int, tmp []byte) (int, error) {
	lenByteOffset := offset
	if offset >= len(msg) {
		return offset, ErrBuf
	}
	offset++
	bs := tmp[:len(s)]
	copy(bs, s)
	for i := 0; i < len(bs); i++ {
		if len(msg) <= offset {
			return offset, ErrBuf
		}
		if bs[i] == '\\' {
			i++
			if i == len(bs) {
				break
			}
			// check for \DDD
			if i+2 < len(bs) && isDigit(bs[i]) && isDigit(bs[i+1]) && isDigit(bs[i+2]) {
				msg[offset] = dddToByte(bs[i:])
				i += 2
			} else if bs[i] == 't' {
				msg[offset] = '\t'
			} else if bs[i] == 'r' {
				msg[offset] = '\r'
			} else if bs[i] == 'n' {
				msg[offset] = '\n'
			} else {
				msg[offset] = bs[i]
			}
		} else {
			msg[offset] = bs[i]
		}
		offset++
	}
	l := offset - lenByteOffset - 1
	if l > 255 {
		return offset, &Error{err: "string exceeded 255 bytes in txt"}
	}
	msg[lenByteOffset] = byte(l)
	return offset, nil
}

func unpackTxt(msg []byte, offset, rdend int) ([]string, int, error) {
	var err error
	var ss []string
	var s string
	for offset < rdend && err == nil {
		s, offset, err = unpackTxtString(msg, offset)
		if err == nil {
			ss = append(ss, s)
		}
	}
	return ss, offset, err
}

func unpackTxtString(msg []byte, offset int) (string, int, error) {
	if offset+1 > len(msg) {
		return "", offset, &Error{err: "overflow unpacking txt"}
	}
	l := int(msg[offset])
	if offset+l+1 > len(msg) {
		return "", offset, &Error{err: "overflow unpacking txt"}
	}
	s := make([]byte, 0, l)
	for _, b := range msg[offset+1 : offset+1+l] {
		switch b {
		case '"', '\\':
			s = append(s, '\\', b)
		case '\t':
			s = append(s, `\t`...)
		case '\r':
			s = append(s, `\r`...)
		case '\n':
			s = append(s, `\n`...)
		default:
			if b < 32 || b > 127 { // unprintable
				var buf [3]byte
				bufs := strconv.AppendInt(buf[:0], int64(b), 10)
				s = append(s, '\\')
				for i := 0; i < 3-len(bufs); i++ {
					s = append(s, '0')
				}
				for _, r := range bufs {
					s = append(s, r)
				}
			} else {
				s = append(s, b)
			}
		}
	}
	offset += 1 + l
	return string(s), offset, nil
}

// Pack a reflect.StructValue into msg.  Struct members can only be uint8, uint16, uint32, string,
// slices and other (often anonymous) structs.
func packStructValue(val reflect.Value, msg []byte, off int, compression map[string]int, compress bool) (off1 int, err error) {
	var txtTmp []byte
	lenmsg := len(msg)
	numfield := val.NumField()
	for i := 0; i < numfield; i++ {
		typefield := val.Type().Field(i)
		if typefield.Tag == `dns:"-"` {
			continue
		}
		switch fv := val.Field(i); fv.Kind() {
		default:
			return lenmsg, &Error{err: "bad kind packing"}
		case reflect.Interface:
			// PrivateRR is the only RR implementation that has interface field.
			// therefore it's expected that this interface would be PrivateRdata
			switch data := fv.Interface().(type) {
			case PrivateRdata:
				n, err := data.Pack(msg[off:])
				if err != nil {
					return lenmsg, err
				}
				off += n
			default:
				return lenmsg, &Error{err: "bad kind interface packing"}
			}
		case reflect.Slice:
			switch typefield.Tag {
			default:
				return lenmsg, &Error{"bad tag packing slice: " + typefield.Tag.Get("dns")}
			case `dns:"domain-name"`:
				for j := 0; j < val.Field(i).Len(); j++ {
					element := val.Field(i).Index(j).String()
					off, err = PackDomainName(element, msg, off, compression, false && compress)
					if err != nil {
						return lenmsg, err
					}
				}
			case `dns:"txt"`:
				if txtTmp == nil {
					txtTmp = make([]byte, 256*4+1)
				}
				off, err = packTxt(fv.Interface().([]string), msg, off, txtTmp)
				if err != nil {
					return lenmsg, err
				}
			case `dns:"opt"`: // edns
				for j := 0; j < val.Field(i).Len(); j++ {
					element := val.Field(i).Index(j).Interface()
					b, e := element.(EDNS0).pack()
					if e != nil {
						return lenmsg, &Error{err: "overflow packing opt"}
					}
					// Option code
					msg[off], msg[off+1] = packUint16(element.(EDNS0).Option())
					// Length
					msg[off+2], msg[off+3] = packUint16(uint16(len(b)))
					off += 4
					if off+len(b) > lenmsg {
						copy(msg[off:], b)
						off = lenmsg
						continue
					}
					// Actual data
					copy(msg[off:off+len(b)], b)
					off += len(b)
				}
			case `dns:"a"`:
				if val.Type().String() == "dns.IPSECKEY" {
					// Field(2) is GatewayType, must be 1
					if val.Field(2).Uint() != 1 {
						continue
					}
				}
				// It must be a slice of 4, even if it is 16, we encode
				// only the first 4
				if off+net.IPv4len > lenmsg {
					return lenmsg, &Error{err: "overflow packing a"}
				}
				switch fv.Len() {
				case net.IPv6len:
					msg[off] = byte(fv.Index(12).Uint())
					msg[off+1] = byte(fv.Index(13).Uint())
					msg[off+2] = byte(fv.Index(14).Uint())
					msg[off+3] = byte(fv.Index(15).Uint())
					off += net.IPv4len
				case net.IPv4len:
					msg[off] = byte(fv.Index(0).Uint())
					msg[off+1] = byte(fv.Index(1).Uint())
					msg[off+2] = byte(fv.Index(2).Uint())
					msg[off+3] = byte(fv.Index(3).Uint())
					off += net.IPv4len
				case 0:
					// Allowed, for dynamic updates
				default:
					return lenmsg, &Error{err: "overflow packing a"}
				}
			case `dns:"aaaa"`:
				if val.Type().String() == "dns.IPSECKEY" {
					// Field(2) is GatewayType, must be 2
					if val.Field(2).Uint() != 2 {
						continue
					}
				}
				if fv.Len() == 0 {
					break
				}
				if fv.Len() > net.IPv6len || off+fv.Len() > lenmsg {
					return lenmsg, &Error{err: "overflow packing aaaa"}
				}
				for j := 0; j < net.IPv6len; j++ {
					msg[off] = byte(fv.Index(j).Uint())
					off++
				}
			case `dns:"wks"`:
				// TODO(miek): this is wrong should be lenrd
				if off == lenmsg {
					break // dyn. updates
				}
				if val.Field(i).Len() == 0 {
					break
				}
				var bitmapbyte uint16
				for j := 0; j < val.Field(i).Len(); j++ {
					serv := uint16((fv.Index(j).Uint()))
					bitmapbyte = uint16(serv / 8)
					if int(bitmapbyte) > lenmsg {
						return lenmsg, &Error{err: "overflow packing wks"}
					}
					bit := uint16(serv) - bitmapbyte*8
					msg[bitmapbyte] = byte(1 << (7 - bit))
				}
				off += int(bitmapbyte)
			case `dns:"nsec"`: // NSEC/NSEC3
				// This is the uint16 type bitmap
				if val.Field(i).Len() == 0 {
					// Do absolutely nothing
					break
				}

				lastwindow := uint16(0)
				length := uint16(0)
				if off+2 > lenmsg {
					return lenmsg, &Error{err: "overflow packing nsecx"}
				}
				for j := 0; j < val.Field(i).Len(); j++ {
					t := uint16((fv.Index(j).Uint()))
					window := uint16(t / 256)
					if lastwindow != window {
						// New window, jump to the new offset
						off += int(length) + 3
						if off > lenmsg {
							return lenmsg, &Error{err: "overflow packing nsecx bitmap"}
						}
					}
					length = (t - window*256) / 8
					bit := t - (window * 256) - (length * 8)
					if off+2+int(length) > lenmsg {
						return lenmsg, &Error{err: "overflow packing nsecx bitmap"}
					}

					// Setting the window #
					msg[off] = byte(window)
					// Setting the octets length
					msg[off+1] = byte(length + 1)
					// Setting the bit value for the type in the right octet
					msg[off+2+int(length)] |= byte(1 << (7 - bit))
					lastwindow = window
				}
				off += 2 + int(length)
				off++
				if off > lenmsg {
					return lenmsg, &Error{err: "overflow packing nsecx bitmap"}
				}
			}
		case reflect.Struct:
			off, err = packStructValue(fv, msg, off, compression, compress)
			if err != nil {
				return lenmsg, err
			}
		case reflect.Uint8:
			if off+1 > lenmsg {
				return lenmsg, &Error{err: "overflow packing uint8"}
			}
			msg[off] = byte(fv.Uint())
			off++
		case reflect.Uint16:
			if off+2 > lenmsg {
				return lenmsg, &Error{err: "overflow packing uint16"}
			}
			i := fv.Uint()
			msg[off] = byte(i >> 8)
			msg[off+1] = byte(i)
			off += 2
		case reflect.Uint32:
			if off+4 > lenmsg {
				return lenmsg, &Error{err: "overflow packing uint32"}
			}
			i := fv.Uint()
			msg[off] = byte(i >> 24)
			msg[off+1] = byte(i >> 16)
			msg[off+2] = byte(i >> 8)
			msg[off+3] = byte(i)
			off += 4
		case reflect.Uint64:
			switch typefield.Tag {
			default:
				if off+8 > lenmsg {
					return lenmsg, &Error{err: "overflow packing uint64"}
				}
				i := fv.Uint()
				msg[off] = byte(i >> 56)
				msg[off+1] = byte(i >> 48)
				msg[off+2] = byte(i >> 40)
				msg[off+3] = byte(i >> 32)
				msg[off+4] = byte(i >> 24)
				msg[off+5] = byte(i >> 16)
				msg[off+6] = byte(i >> 8)
				msg[off+7] = byte(i)
				off += 8
			case `dns:"uint48"`:
				// Used in TSIG, where it stops at 48 bits, so we discard the upper 16
				if off+6 > lenmsg {
					return lenmsg, &Error{err: "overflow packing uint64 as uint48"}
				}
				i := fv.Uint()
				msg[off] = byte(i >> 40)
				msg[off+1] = byte(i >> 32)
				msg[off+2] = byte(i >> 24)
				msg[off+3] = byte(i >> 16)
				msg[off+4] = byte(i >> 8)
				msg[off+5] = byte(i)
				off += 6
			}
		case reflect.String:
			// There are multiple string encodings.
			// The tag distinguishes ordinary strings from domain names.
			s := fv.String()
			switch typefield.Tag {
			default:
				return lenmsg, &Error{"bad tag packing string: " + typefield.Tag.Get("dns")}
			case `dns:"base64"`:
				b64, e := fromBase64([]byte(s))
				if e != nil {
					return lenmsg, e
				}
				copy(msg[off:off+len(b64)], b64)
				off += len(b64)
			case `dns:"domain-name"`:
				if val.Type().String() == "dns.IPSECKEY" {
					// Field(2) is GatewayType, 1 and 2 or used for addresses
					x := val.Field(2).Uint()
					if x == 1 || x == 2 {
						continue
					}
				}
				if off, err = PackDomainName(s, msg, off, compression, false && compress); err != nil {
					return lenmsg, err
				}
			case `dns:"cdomain-name"`:
				if off, err = PackDomainName(s, msg, off, compression, true && compress); err != nil {
					return lenmsg, err
				}
			case `dns:"size-base32"`:
				// This is purely for NSEC3 atm, the previous byte must
				// holds the length of the encoded string. As NSEC3
				// is only defined to SHA1, the hashlength is 20 (160 bits)
				msg[off-1] = 20
				fallthrough
			case `dns:"base32"`:
				b32, e := fromBase32([]byte(s))
				if e != nil {
					return lenmsg, e
				}
				copy(msg[off:off+len(b32)], b32)
				off += len(b32)
			case `dns:"size-hex"`:
				fallthrough
			case `dns:"hex"`:
				// There is no length encoded here
				h, e := hex.DecodeString(s)
				if e != nil {
					return lenmsg, e
				}
				if off+hex.DecodedLen(len(s)) > lenmsg {
					return lenmsg, &Error{err: "overflow packing hex"}
				}
				copy(msg[off:off+hex.DecodedLen(len(s))], h)
				off += hex.DecodedLen(len(s))
			case `dns:"size"`:
				// the size is already encoded in the RR, we can safely use the
				// length of string. String is RAW (not encoded in hex, nor base64)
				copy(msg[off:off+len(s)], s)
				off += len(s)
			case `dns:"txt"`:
				fallthrough
			case "":
				if txtTmp == nil {
					txtTmp = make([]byte, 256*4+1)
				}
				off, err = packTxtString(fv.String(), msg, off, txtTmp)
				if err != nil {
					return lenmsg, err
				}
			}
		}
	}
	return off, nil
}

func structValue(any interface{}) reflect.Value {
	return reflect.ValueOf(any).Elem()
}

// PackStruct packs any structure to wire format.
func PackStruct(any interface{}, msg []byte, off int) (off1 int, err error) {
	off, err = packStructValue(structValue(any), msg, off, nil, false)
	return off, err
}

func packStructCompress(any interface{}, msg []byte, off int, compression map[string]int, compress bool) (off1 int, err error) {
	off, err = packStructValue(structValue(any), msg, off, compression, compress)
	return off, err
}

// TODO(miek): Fix use of rdlength here

// Unpack a reflect.StructValue from msg.
// Same restrictions as packStructValue.
func unpackStructValue(val reflect.Value, msg []byte, off int) (off1 int, err error) {
	var lenrd int
	lenmsg := len(msg)
	for i := 0; i < val.NumField(); i++ {
		if lenrd != 0 && lenrd == off {
			break
		}
		if off > lenmsg {
			return lenmsg, &Error{"bad offset unpacking"}
		}
		switch fv := val.Field(i); fv.Kind() {
		default:
			return lenmsg, &Error{err: "bad kind unpacking"}
		case reflect.Interface:
			// PrivateRR is the only RR implementation that has interface field.
			// therefore it's expected that this interface would be PrivateRdata
			switch data := fv.Interface().(type) {
			case PrivateRdata:
				n, err := data.Unpack(msg[off:lenrd])
				if err != nil {
					return lenmsg, err
				}
				off += n
			default:
				return lenmsg, &Error{err: "bad kind interface unpacking"}
			}
		case reflect.Slice:
			switch val.Type().Field(i).Tag {
			default:
				return lenmsg, &Error{"bad tag unpacking slice: " + val.Type().Field(i).Tag.Get("dns")}
			case `dns:"domain-name"`:
				// HIP record slice of name (or none)
				var servers []string
				var s string
				for off < lenrd {
					s, off, err = UnpackDomainName(msg, off)
					if err != nil {
						return lenmsg, err
					}
					servers = append(servers, s)
				}
				fv.Set(reflect.ValueOf(servers))
			case `dns:"txt"`:
				if off == lenmsg || lenrd == off {
					break
				}
				var txt []string
				txt, off, err = unpackTxt(msg, off, lenrd)
				if err != nil {
					return lenmsg, err
				}
				fv.Set(reflect.ValueOf(txt))
			case `dns:"opt"`: // edns0
				if off == lenrd {
					// This is an EDNS0 (OPT Record) with no rdata
					// We can safely return here.
					break
				}
				var edns []EDNS0
			Option:
				code := uint16(0)
				if off+2 > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking opt"}
				}
				code, off = unpackUint16(msg, off)
				optlen, off1 := unpackUint16(msg, off)
				if off1+int(optlen) > lenrd {
					return lenmsg, &Error{err: "overflow unpacking opt"}
				}
				switch code {
				case EDNS0NSID:
					e := new(EDNS0_NSID)
					if err := e.unpack(msg[off1 : off1+int(optlen)]); err != nil {
						return lenmsg, err
					}
					edns = append(edns, e)
					off = off1 + int(optlen)
				case EDNS0SUBNET, EDNS0SUBNETDRAFT:
					e := new(EDNS0_SUBNET)
					if err := e.unpack(msg[off1 : off1+int(optlen)]); err != nil {
						return lenmsg, err
					}
					edns = append(edns, e)
					off = off1 + int(optlen)
					if code == EDNS0SUBNETDRAFT {
						e.DraftOption = true
					}
				case EDNS0UL:
					e := new(EDNS0_UL)
					if err := e.unpack(msg[off1 : off1+int(optlen)]); err != nil {
						return lenmsg, err
					}
					edns = append(edns, e)
					off = off1 + int(optlen)
				case EDNS0LLQ:
					e := new(EDNS0_LLQ)
					if err := e.unpack(msg[off1 : off1+int(optlen)]); err != nil {
						return lenmsg, err
					}
					edns = append(edns, e)
					off = off1 + int(optlen)
				case EDNS0DAU:
					e := new(EDNS0_DAU)
					if err := e.unpack(msg[off1 : off1+int(optlen)]); err != nil {
						return lenmsg, err
					}
					edns = append(edns, e)
					off = off1 + int(optlen)
				case EDNS0DHU:
					e := new(EDNS0_DHU)
					if err := e.unpack(msg[off1 : off1+int(optlen)]); err != nil {
						return lenmsg, err
					}
					edns = append(edns, e)
					off = off1 + int(optlen)
				case EDNS0N3U:
					e := new(EDNS0_N3U)
					if err := e.unpack(msg[off1 : off1+int(optlen)]); err != nil {
						return lenmsg, err
					}
					edns = append(edns, e)
					off = off1 + int(optlen)
				default:
					e := new(EDNS0_LOCAL)
					e.Code = code
					if err := e.unpack(msg[off1 : off1+int(optlen)]); err != nil {
						return lenmsg, err
					}
					edns = append(edns, e)
					off = off1 + int(optlen)
				}
				if off < lenrd {
					goto Option
				}
				fv.Set(reflect.ValueOf(edns))
			case `dns:"a"`:
				if val.Type().String() == "dns.IPSECKEY" {
					// Field(2) is GatewayType, must be 1
					if val.Field(2).Uint() != 1 {
						continue
					}
				}
				if off == lenrd {
					break // dyn. update
				}
				if off+net.IPv4len > lenrd || off+net.IPv4len > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking a"}
				}
				fv.Set(reflect.ValueOf(net.IPv4(msg[off], msg[off+1], msg[off+2], msg[off+3])))
				off += net.IPv4len
			case `dns:"aaaa"`:
				if val.Type().String() == "dns.IPSECKEY" {
					// Field(2) is GatewayType, must be 2
					if val.Field(2).Uint() != 2 {
						continue
					}
				}
				if off == lenrd {
					break
				}
				if off+net.IPv6len > lenrd || off+net.IPv6len > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking aaaa"}
				}
				fv.Set(reflect.ValueOf(net.IP{msg[off], msg[off+1], msg[off+2], msg[off+3], msg[off+4],
					msg[off+5], msg[off+6], msg[off+7], msg[off+8], msg[off+9], msg[off+10],
					msg[off+11], msg[off+12], msg[off+13], msg[off+14], msg[off+15]}))
				off += net.IPv6len
			case `dns:"wks"`:
				// Rest of the record is the bitmap
				var serv []uint16
				j := 0
				for off < lenrd {
					if off+1 > lenmsg {
						return lenmsg, &Error{err: "overflow unpacking wks"}
					}
					b := msg[off]
					// Check the bits one by one, and set the type
					if b&0x80 == 0x80 {
						serv = append(serv, uint16(j*8+0))
					}
					if b&0x40 == 0x40 {
						serv = append(serv, uint16(j*8+1))
					}
					if b&0x20 == 0x20 {
						serv = append(serv, uint16(j*8+2))
					}
					if b&0x10 == 0x10 {
						serv = append(serv, uint16(j*8+3))
					}
					if b&0x8 == 0x8 {
						serv = append(serv, uint16(j*8+4))
					}
					if b&0x4 == 0x4 {
						serv = append(serv, uint16(j*8+5))
					}
					if b&0x2 == 0x2 {
						serv = append(serv, uint16(j*8+6))
					}
					if b&0x1 == 0x1 {
						serv = append(serv, uint16(j*8+7))
					}
					j++
					off++
				}
				fv.Set(reflect.ValueOf(serv))
			case `dns:"nsec"`: // NSEC/NSEC3
				if off == lenrd {
					break
				}
				// Rest of the record is the type bitmap
				if off+2 > lenrd || off+2 > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking nsecx"}
				}
				var nsec []uint16
				length := 0
				window := 0
				for off+2 < lenrd {
					window = int(msg[off])
					length = int(msg[off+1])
					//println("off, windows, length, end", off, window, length, endrr)
					if length == 0 {
						// A length window of zero is strange. If there
						// the window should not have been specified. Bail out
						// println("dns: length == 0 when unpacking NSEC")
						return lenmsg, &Error{err: "overflow unpacking nsecx"}
					}
					if length > 32 {
						return lenmsg, &Error{err: "overflow unpacking nsecx"}
					}

					// Walk the bytes in the window - and check the bit settings...
					off += 2
					for j := 0; j < length; j++ {
						if off+j+1 > lenmsg {
							return lenmsg, &Error{err: "overflow unpacking nsecx"}
						}
						b := msg[off+j]
						// Check the bits one by one, and set the type
						if b&0x80 == 0x80 {
							nsec = append(nsec, uint16(window*256+j*8+0))
						}
						if b&0x40 == 0x40 {
							nsec = append(nsec, uint16(window*256+j*8+1))
						}
						if b&0x20 == 0x20 {
							nsec = append(nsec, uint16(window*256+j*8+2))
						}
						if b&0x10 == 0x10 {
							nsec = append(nsec, uint16(window*256+j*8+3))
						}
						if b&0x8 == 0x8 {
							nsec = append(nsec, uint16(window*256+j*8+4))
						}
						if b&0x4 == 0x4 {
							nsec = append(nsec, uint16(window*256+j*8+5))
						}
						if b&0x2 == 0x2 {
							nsec = append(nsec, uint16(window*256+j*8+6))
						}
						if b&0x1 == 0x1 {
							nsec = append(nsec, uint16(window*256+j*8+7))
						}
					}
					off += length
				}
				fv.Set(reflect.ValueOf(nsec))
			}
		case reflect.Struct:
			off, err = unpackStructValue(fv, msg, off)
			if err != nil {
				return lenmsg, err
			}
			if val.Type().Field(i).Name == "Hdr" {
				lenrd = off + int(val.FieldByName("Hdr").FieldByName("Rdlength").Uint())
			}
		case reflect.Uint8:
			if off == lenmsg {
				break
			}
			if off+1 > lenmsg {
				return lenmsg, &Error{err: "overflow unpacking uint8"}
			}
			fv.SetUint(uint64(uint8(msg[off])))
			off++
		case reflect.Uint16:
			if off == lenmsg {
				break
			}
			var i uint16
			if off+2 > lenmsg {
				return lenmsg, &Error{err: "overflow unpacking uint16"}
			}
			i, off = unpackUint16(msg, off)
			fv.SetUint(uint64(i))
		case reflect.Uint32:
			if off == lenmsg {
				break
			}
			if off+4 > lenmsg {
				return lenmsg, &Error{err: "overflow unpacking uint32"}
			}
			fv.SetUint(uint64(uint32(msg[off])<<24 | uint32(msg[off+1])<<16 | uint32(msg[off+2])<<8 | uint32(msg[off+3])))
			off += 4
		case reflect.Uint64:
			switch val.Type().Field(i).Tag {
			default:
				if off+8 > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking uint64"}
				}
				fv.SetUint(uint64(uint64(msg[off])<<56 | uint64(msg[off+1])<<48 | uint64(msg[off+2])<<40 |
					uint64(msg[off+3])<<32 | uint64(msg[off+4])<<24 | uint64(msg[off+5])<<16 | uint64(msg[off+6])<<8 | uint64(msg[off+7])))
				off += 8
			case `dns:"uint48"`:
				// Used in TSIG where the last 48 bits are occupied, so for now, assume a uint48 (6 bytes)
				if off+6 > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking uint64 as uint48"}
				}
				fv.SetUint(uint64(uint64(msg[off])<<40 | uint64(msg[off+1])<<32 | uint64(msg[off+2])<<24 | uint64(msg[off+3])<<16 |
					uint64(msg[off+4])<<8 | uint64(msg[off+5])))
				off += 6
			}
		case reflect.String:
			var s string
			if off == lenmsg {
				break
			}
			switch val.Type().Field(i).Tag {
			default:
				return lenmsg, &Error{"bad tag unpacking string: " + val.Type().Field(i).Tag.Get("dns")}
			case `dns:"hex"`:
				hexend := lenrd
				if val.FieldByName("Hdr").FieldByName("Rrtype").Uint() == uint64(TypeHIP) {
					hexend = off + int(val.FieldByName("HitLength").Uint())
				}
				if hexend > lenrd || hexend > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking hex"}
				}
				s = hex.EncodeToString(msg[off:hexend])
				off = hexend
			case `dns:"base64"`:
				// Rest of the RR is base64 encoded value
				b64end := lenrd
				if val.FieldByName("Hdr").FieldByName("Rrtype").Uint() == uint64(TypeHIP) {
					b64end = off + int(val.FieldByName("PublicKeyLength").Uint())
				}
				if b64end > lenrd || b64end > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking base64"}
				}
				s = toBase64(msg[off:b64end])
				off = b64end
			case `dns:"cdomain-name"`:
				fallthrough
			case `dns:"domain-name"`:
				if val.Type().String() == "dns.IPSECKEY" {
					// Field(2) is GatewayType, 1 and 2 or used for addresses
					x := val.Field(2).Uint()
					if x == 1 || x == 2 {
						continue
					}
				}
				if off == lenmsg {
					// zero rdata foo, OK for dyn. updates
					break
				}
				s, off, err = UnpackDomainName(msg, off)
				if err != nil {
					return lenmsg, err
				}
			case `dns:"size-base32"`:
				var size int
				switch val.Type().Name() {
				case "NSEC3":
					switch val.Type().Field(i).Name {
					case "NextDomain":
						name := val.FieldByName("HashLength")
						size = int(name.Uint())
					}
				}
				if off+size > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking base32"}
				}
				s = toBase32(msg[off : off+size])
				off += size
			case `dns:"size-hex"`:
				// a "size" string, but it must be encoded in hex in the string
				var size int
				switch val.Type().Name() {
				case "NSEC3":
					switch val.Type().Field(i).Name {
					case "Salt":
						name := val.FieldByName("SaltLength")
						size = int(name.Uint())
					case "NextDomain":
						name := val.FieldByName("HashLength")
						size = int(name.Uint())
					}
				case "TSIG":
					switch val.Type().Field(i).Name {
					case "MAC":
						name := val.FieldByName("MACSize")
						size = int(name.Uint())
					case "OtherData":
						name := val.FieldByName("OtherLen")
						size = int(name.Uint())
					}
				}
				if off+size > lenmsg {
					return lenmsg, &Error{err: "overflow unpacking hex"}
				}
				s = hex.EncodeToString(msg[off : off+size])
				off += size
			case `dns:"txt"`:
				fallthrough
			case "":
				s, off, err = unpackTxtString(msg, off)
			}
			fv.SetString(s)
		}
	}
	return off, nil
}

// Helpers for dealing with escaped bytes
func isDigit(b byte) bool { return b >= '0' && b <= '9' }

func dddToByte(s []byte) byte {
	return byte((s[0]-'0')*100 + (s[1]-'0')*10 + (s[2] - '0'))
}

// UnpackStruct unpacks a binary message from offset off to the interface
// value given.
func UnpackStruct(any interface{}, msg []byte, off int) (int, error) {
	return unpackStructValue(structValue(any), msg, off)
}

// Helper function for packing and unpacking
func intToBytes(i *big.Int, length int) []byte {
	buf := i.Bytes()
	if len(buf) < length {
		b := make([]byte, length)
		copy(b[length-len(buf):], buf)
		return b
	}
	return buf
}

func unpackUint16(msg []byte, off int) (uint16, int) {
	return uint16(msg[off])<<8 | uint16(msg[off+1]), off + 2
}

func packUint16(i uint16) (byte, byte) {
	return byte(i >> 8), byte(i)
}

func toBase32(b []byte) string {
	return base32.HexEncoding.EncodeToString(b)
}

func fromBase32(s []byte) (buf []byte, err error) {
	buflen := base32.HexEncoding.DecodedLen(len(s))
	buf = make([]byte, buflen)
	n, err := base32.HexEncoding.Decode(buf, s)
	buf = buf[:n]
	return
}

func toBase64(b []byte) string {
	return base64.StdEncoding.EncodeToString(b)
}

func fromBase64(s []byte) (buf []byte, err error) {
	buflen := base64.StdEncoding.DecodedLen(len(s))
	buf = make([]byte, buflen)
	n, err := base64.StdEncoding.Decode(buf, s)
	buf = buf[:n]
	return
}

// PackRR packs a resource record rr into msg[off:].
// See PackDomainName for documentation about the compression.
func PackRR(rr RR, msg []byte, off int, compression map[string]int, compress bool) (off1 int, err error) {
	if rr == nil {
		return len(msg), &Error{err: "nil rr"}
	}

	off1, err = packStructCompress(rr, msg, off, compression, compress)
	if err != nil {
		return len(msg), err
	}
	if rawSetRdlength(msg, off, off1) {
		return off1, nil
	}
	return off, ErrRdata
}

// UnpackRR unpacks msg[off:] into an RR.
func UnpackRR(msg []byte, off int) (rr RR, off1 int, err error) {
	// unpack just the header, to find the rr type and length
	var h RR_Header
	off0 := off
	if off, err = UnpackStruct(&h, msg, off); err != nil {
		return nil, len(msg), err
	}
	end := off + int(h.Rdlength)
	// make an rr of that type and re-unpack.
	mk, known := typeToRR[h.Rrtype]
	if !known {
		rr = new(RFC3597)
	} else {
		rr = mk()
	}
	off, err = UnpackStruct(rr, msg, off0)
	if off != end {
		return &h, end, &Error{err: "bad rdlength"}
	}
	return rr, off, err
}

// Reverse a map
func reverseInt8(m map[uint8]string) map[string]uint8 {
	n := make(map[string]uint8)
	for u, s := range m {
		n[s] = u
	}
	return n
}

func reverseInt16(m map[uint16]string) map[string]uint16 {
	n := make(map[string]uint16)
	for u, s := range m {
		n[s] = u
	}
	return n
}

func reverseInt(m map[int]string) map[string]int {
	n := make(map[string]int)
	for u, s := range m {
		n[s] = u
	}
	return n
}

// Convert a MsgHdr to a string, with dig-like headers:
//
//;; opcode: QUERY, status: NOERROR, id: 48404
//
//;; flags: qr aa rd ra;
func (h *MsgHdr) String() string {
	if h == nil {
		return "<nil> MsgHdr"
	}

	s := ";; opcode: " + OpcodeToString[h.Opcode]
	s += ", status: " + RcodeToString[h.Rcode]
	s += ", id: " + strconv.Itoa(int(h.Id)) + "\n"

	s += ";; flags:"
	if h.Response {
		s += " qr"
	}
	if h.Authoritative {
		s += " aa"
	}
	if h.Truncated {
		s += " tc"
	}
	if h.RecursionDesired {
		s += " rd"
	}
	if h.RecursionAvailable {
		s += " ra"
	}
	if h.Zero { // Hmm
		s += " z"
	}
	if h.AuthenticatedData {
		s += " ad"
	}
	if h.CheckingDisabled {
		s += " cd"
	}

	s += ";"
	return s
}

// Pack packs a Msg: it is converted to to wire format.
// If the dns.Compress is true the message will be in compressed wire format.
func (dns *Msg) Pack() (msg []byte, err error) {
	return dns.PackBuffer(nil)
}

// PackBuffer packs a Msg, using the given buffer buf. If buf is too small
// a new buffer is allocated.
func (dns *Msg) PackBuffer(buf []byte) (msg []byte, err error) {
	var dh Header
	var compression map[string]int
	if dns.Compress {
		compression = make(map[string]int) // Compression pointer mappings
	}

	if dns.Rcode < 0 || dns.Rcode > 0xFFF {
		return nil, ErrRcode
	}
	if dns.Rcode > 0xF {
		// Regular RCODE field is 4 bits
		opt := dns.IsEdns0()
		if opt == nil {
			return nil, ErrExtendedRcode
		}
		opt.SetExtendedRcode(uint8(dns.Rcode >> 4))
		dns.Rcode &= 0xF
	}

	// Convert convenient Msg into wire-like Header.
	dh.Id = dns.Id
	dh.Bits = uint16(dns.Opcode)<<11 | uint16(dns.Rcode)
	if dns.Response {
		dh.Bits |= _QR
	}
	if dns.Authoritative {
		dh.Bits |= _AA
	}
	if dns.Truncated {
		dh.Bits |= _TC
	}
	if dns.RecursionDesired {
		dh.Bits |= _RD
	}
	if dns.RecursionAvailable {
		dh.Bits |= _RA
	}
	if dns.Zero {
		dh.Bits |= _Z
	}
	if dns.AuthenticatedData {
		dh.Bits |= _AD
	}
	if dns.CheckingDisabled {
		dh.Bits |= _CD
	}

	// Prepare variable sized arrays.
	question := dns.Question
	answer := dns.Answer
	ns := dns.Ns
	extra := dns.Extra

	dh.Qdcount = uint16(len(question))
	dh.Ancount = uint16(len(answer))
	dh.Nscount = uint16(len(ns))
	dh.Arcount = uint16(len(extra))

	// We need the uncompressed length here, because we first pack it and then compress it.
	msg = buf
	compress := dns.Compress
	dns.Compress = false
	if packLen := dns.Len() + 1; len(msg) < packLen {
		msg = make([]byte, packLen)
	}
	dns.Compress = compress

	// Pack it in: header and then the pieces.
	off := 0
	off, err = packStructCompress(&dh, msg, off, compression, dns.Compress)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(question); i++ {
		off, err = packStructCompress(&question[i], msg, off, compression, dns.Compress)
		if err != nil {
			return nil, err
		}
	}
	for i := 0; i < len(answer); i++ {
		off, err = PackRR(answer[i], msg, off, compression, dns.Compress)
		if err != nil {
			return nil, err
		}
	}
	for i := 0; i < len(ns); i++ {
		off, err = PackRR(ns[i], msg, off, compression, dns.Compress)
		if err != nil {
			return nil, err
		}
	}
	for i := 0; i < len(extra); i++ {
		off, err = PackRR(extra[i], msg, off, compression, dns.Compress)
		if err != nil {
			return nil, err
		}
	}
	return msg[:off], nil
}

// Unpack unpacks a binary message to a Msg structure.
func (dns *Msg) Unpack(msg []byte) (err error) {
	// Header.
	var dh Header
	off := 0
	if off, err = UnpackStruct(&dh, msg, off); err != nil {
		return err
	}
	dns.Id = dh.Id
	dns.Response = (dh.Bits & _QR) != 0
	dns.Opcode = int(dh.Bits>>11) & 0xF
	dns.Authoritative = (dh.Bits & _AA) != 0
	dns.Truncated = (dh.Bits & _TC) != 0
	dns.RecursionDesired = (dh.Bits & _RD) != 0
	dns.RecursionAvailable = (dh.Bits & _RA) != 0
	dns.Zero = (dh.Bits & _Z) != 0
	dns.AuthenticatedData = (dh.Bits & _AD) != 0
	dns.CheckingDisabled = (dh.Bits & _CD) != 0
	dns.Rcode = int(dh.Bits & 0xF)

	// Arrays.
	dns.Question = make([]Question, dh.Qdcount)
	dns.Answer = make([]RR, dh.Ancount)
	dns.Ns = make([]RR, dh.Nscount)
	dns.Extra = make([]RR, dh.Arcount)

	for i := 0; i < len(dns.Question); i++ {
		off, err = UnpackStruct(&dns.Question[i], msg, off)
		if err != nil {
			return err
		}
	}
	// If we see a TC bit being set we return here, without
	// an error, because technically it isn't an error. So return
	// without parsing the potentially corrupt packet and hitting an error.
	// TODO(miek): this isn't the best strategy!
	if dns.Truncated {
		dns.Answer = nil
		dns.Ns = nil
		dns.Extra = nil
		return nil
	}
	for i := 0; i < len(dns.Answer); i++ {
		dns.Answer[i], off, err = UnpackRR(msg, off)
		if err != nil {
			return err
		}
	}
	for i := 0; i < len(dns.Ns); i++ {
		dns.Ns[i], off, err = UnpackRR(msg, off)
		if err != nil {
			return err
		}
	}
	for i := 0; i < len(dns.Extra); i++ {
		dns.Extra[i], off, err = UnpackRR(msg, off)
		if err != nil {
			return err
		}
	}
	if off != len(msg) {
		// TODO(miek) make this an error?
		// use PackOpt to let people tell how detailed the error reporting should be?
		// println("dns: extra bytes in dns packet", off, "<", len(msg))
	}
	return nil
}

// Convert a complete message to a string with dig-like output.
func (dns *Msg) String() string {
	if dns == nil {
		return "<nil> MsgHdr"
	}
	s := dns.MsgHdr.String() + " "
	s += "QUERY: " + strconv.Itoa(len(dns.Question)) + ", "
	s += "ANSWER: " + strconv.Itoa(len(dns.Answer)) + ", "
	s += "AUTHORITY: " + strconv.Itoa(len(dns.Ns)) + ", "
	s += "ADDITIONAL: " + strconv.Itoa(len(dns.Extra)) + "\n"
	if len(dns.Question) > 0 {
		s += "\n;; QUESTION SECTION:\n"
		for i := 0; i < len(dns.Question); i++ {
			s += dns.Question[i].String() + "\n"
		}
	}
	if len(dns.Answer) > 0 {
		s += "\n;; ANSWER SECTION:\n"
		for i := 0; i < len(dns.Answer); i++ {
			if dns.Answer[i] != nil {
				s += dns.Answer[i].String() + "\n"
			}
		}
	}
	if len(dns.Ns) > 0 {
		s += "\n;; AUTHORITY SECTION:\n"
		for i := 0; i < len(dns.Ns); i++ {
			if dns.Ns[i] != nil {
				s += dns.Ns[i].String() + "\n"
			}
		}
	}
	if len(dns.Extra) > 0 {
		s += "\n;; ADDITIONAL SECTION:\n"
		for i := 0; i < len(dns.Extra); i++ {
			if dns.Extra[i] != nil {
				s += dns.Extra[i].String() + "\n"
			}
		}
	}
	return s
}

// Len returns the message length when in (un)compressed wire format.
// If dns.Compress is true compression it is taken into account. Len()
// is provided to be a faster way to get the size of the resulting packet,
// than packing it, measuring the size and discarding the buffer.
func (dns *Msg) Len() int {
	// We always return one more than needed.
	l := 12 // Message header is always 12 bytes
	var compression map[string]int
	if dns.Compress {
		compression = make(map[string]int)
	}
	for i := 0; i < len(dns.Question); i++ {
		l += dns.Question[i].len()
		if dns.Compress {
			compressionLenHelper(compression, dns.Question[i].Name)
		}
	}
	for i := 0; i < len(dns.Answer); i++ {
		l += dns.Answer[i].len()
		if dns.Compress {
			k, ok := compressionLenSearch(compression, dns.Answer[i].Header().Name)
			if ok {
				l += 1 - k
			}
			compressionLenHelper(compression, dns.Answer[i].Header().Name)
			k, ok = compressionLenSearchType(compression, dns.Answer[i])
			if ok {
				l += 1 - k
			}
			compressionLenHelperType(compression, dns.Answer[i])
		}
	}
	for i := 0; i < len(dns.Ns); i++ {
		l += dns.Ns[i].len()
		if dns.Compress {
			k, ok := compressionLenSearch(compression, dns.Ns[i].Header().Name)
			if ok {
				l += 1 - k
			}
			compressionLenHelper(compression, dns.Ns[i].Header().Name)
			k, ok = compressionLenSearchType(compression, dns.Ns[i])
			if ok {
				l += 1 - k
			}
			compressionLenHelperType(compression, dns.Ns[i])
		}
	}
	for i := 0; i < len(dns.Extra); i++ {
		l += dns.Extra[i].len()
		if dns.Compress {
			k, ok := compressionLenSearch(compression, dns.Extra[i].Header().Name)
			if ok {
				l += 1 - k
			}
			compressionLenHelper(compression, dns.Extra[i].Header().Name)
			k, ok = compressionLenSearchType(compression, dns.Extra[i])
			if ok {
				l += 1 - k
			}
			compressionLenHelperType(compression, dns.Extra[i])
		}
	}
	return l
}

// Put the parts of the name in the compression map.
func compressionLenHelper(c map[string]int, s string) {
	pref := ""
	lbs := Split(s)
	for j := len(lbs) - 1; j >= 0; j-- {
		pref = s[lbs[j]:]
		if _, ok := c[pref]; !ok {
			c[pref] = len(pref)
		}
	}
}

// Look for each part in the compression map and returns its length,
// keep on searching so we get the longest match.
func compressionLenSearch(c map[string]int, s string) (int, bool) {
	off := 0
	end := false
	if s == "" { // don't bork on bogus data
		return 0, false
	}
	for {
		if _, ok := c[s[off:]]; ok {
			return len(s[off:]), true
		}
		if end {
			break
		}
		off, end = NextLabel(s, off)
	}
	return 0, false
}

// TODO(miek): should add all types, because the all can be *used* for compression.
func compressionLenHelperType(c map[string]int, r RR) {
	switch x := r.(type) {
	case *NS:
		compressionLenHelper(c, x.Ns)
	case *MX:
		compressionLenHelper(c, x.Mx)
	case *CNAME:
		compressionLenHelper(c, x.Target)
	case *PTR:
		compressionLenHelper(c, x.Ptr)
	case *SOA:
		compressionLenHelper(c, x.Ns)
		compressionLenHelper(c, x.Mbox)
	case *MB:
		compressionLenHelper(c, x.Mb)
	case *MG:
		compressionLenHelper(c, x.Mg)
	case *MR:
		compressionLenHelper(c, x.Mr)
	case *MF:
		compressionLenHelper(c, x.Mf)
	case *MD:
		compressionLenHelper(c, x.Md)
	case *RT:
		compressionLenHelper(c, x.Host)
	case *MINFO:
		compressionLenHelper(c, x.Rmail)
		compressionLenHelper(c, x.Email)
	case *AFSDB:
		compressionLenHelper(c, x.Hostname)
	}
}

// Only search on compressing these types.
func compressionLenSearchType(c map[string]int, r RR) (int, bool) {
	switch x := r.(type) {
	case *NS:
		return compressionLenSearch(c, x.Ns)
	case *MX:
		return compressionLenSearch(c, x.Mx)
	case *CNAME:
		return compressionLenSearch(c, x.Target)
	case *PTR:
		return compressionLenSearch(c, x.Ptr)
	case *SOA:
		k, ok := compressionLenSearch(c, x.Ns)
		k1, ok1 := compressionLenSearch(c, x.Mbox)
		if !ok && !ok1 {
			return 0, false
		}
		return k + k1, true
	case *MB:
		return compressionLenSearch(c, x.Mb)
	case *MG:
		return compressionLenSearch(c, x.Mg)
	case *MR:
		return compressionLenSearch(c, x.Mr)
	case *MF:
		return compressionLenSearch(c, x.Mf)
	case *MD:
		return compressionLenSearch(c, x.Md)
	case *RT:
		return compressionLenSearch(c, x.Host)
	case *MINFO:
		k, ok := compressionLenSearch(c, x.Rmail)
		k1, ok1 := compressionLenSearch(c, x.Email)
		if !ok && !ok1 {
			return 0, false
		}
		return k + k1, true
	case *AFSDB:
		return compressionLenSearch(c, x.Hostname)
	}
	return 0, false
}

// id returns a 16 bits random number to be used as a
// message id. The random provided should be good enough.
func id() uint16 {
	return uint16(rand.Int()) ^ uint16(time.Now().Nanosecond())
}

// Copy returns a new RR which is a deep-copy of r.
func Copy(r RR) RR {
	r1 := r.copy()
	return r1
}

// Copy returns a new *Msg which is a deep-copy of dns.
func (dns *Msg) Copy() *Msg {
	return dns.CopyTo(new(Msg))
}

// CopyTo copies the contents to the provided message using a deep-copy and returns the copy.
func (dns *Msg) CopyTo(r1 *Msg) *Msg {
	r1.MsgHdr = dns.MsgHdr
	r1.Compress = dns.Compress

	if len(dns.Question) > 0 {
		r1.Question = make([]Question, len(dns.Question))
		copy(r1.Question, dns.Question) // TODO(miek): Question is an immutable value, ok to do a shallow-copy
	}

	rrArr := make([]RR, len(dns.Answer)+len(dns.Ns)+len(dns.Extra))
	var rri int

	if len(dns.Answer) > 0 {
		rrbegin := rri
		for i := 0; i < len(dns.Answer); i++ {
			rrArr[rri] = dns.Answer[i].copy()
			rri++
		}
		r1.Answer = rrArr[rrbegin:rri:rri]
	}

	if len(dns.Ns) > 0 {
		rrbegin := rri
		for i := 0; i < len(dns.Ns); i++ {
			rrArr[rri] = dns.Ns[i].copy()
			rri++
		}
		r1.Ns = rrArr[rrbegin:rri:rri]
	}

	if len(dns.Extra) > 0 {
		rrbegin := rri
		for i := 0; i < len(dns.Extra); i++ {
			rrArr[rri] = dns.Extra[i].copy()
			rri++
		}
		r1.Extra = rrArr[rrbegin:rri:rri]
	}

	return r1
}
