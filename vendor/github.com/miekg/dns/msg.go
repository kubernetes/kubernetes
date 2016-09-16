// DNS packet assembly, see RFC 1035. Converting from - Unpack() -
// and to - Pack() - wire format.
// All the packers and unpackers take a (msg []byte, off int)
// and return (off1 int, ok bool).  If they return ok==false, they
// also return off1==len(msg), so that the next unpacker will
// also fail.  This lets us avoid checks of ok until the end of a
// packing sequence.

package dns

//go:generate go run msg_generate.go

import (
	crand "crypto/rand"
	"encoding/binary"
	"math/big"
	"math/rand"
	"strconv"
)

func init() {
	// Initialize default math/rand source using crypto/rand to provide better
	// security without the performance trade-off.
	buf := make([]byte, 8)
	_, err := crand.Read(buf)
	if err != nil {
		// Failed to read from cryptographic source, fallback to default initial
		// seed (1) by returning early
		return
	}
	seed := binary.BigEndian.Uint64(buf)
	rand.Seed(int64(seed))
}

const maxCompressionOffset = 2 << 13 // We have 14 bits for the compression pointer

var (
	ErrAlg           error = &Error{err: "bad algorithm"}                  // ErrAlg indicates an error with the (DNSSEC) algorithm.
	ErrAuth          error = &Error{err: "bad authentication"}             // ErrAuth indicates an error in the TSIG authentication.
	ErrBuf           error = &Error{err: "buffer size too small"}          // ErrBuf indicates that the buffer used it too small for the message.
	ErrConnEmpty     error = &Error{err: "conn has no connection"}         // ErrConnEmpty indicates a connection is being uses before it is initialized.
	ErrExtendedRcode error = &Error{err: "bad extended rcode"}             // ErrExtendedRcode ...
	ErrFqdn          error = &Error{err: "domain must be fully qualified"} // ErrFqdn indicates that a domain name does not have a closing dot.
	ErrId            error = &Error{err: "id mismatch"}                    // ErrId indicates there is a mismatch with the message's ID.
	ErrKeyAlg        error = &Error{err: "bad key algorithm"}              // ErrKeyAlg indicates that the algorithm in the key is not valid.
	ErrKey           error = &Error{err: "bad key"}
	ErrKeySize       error = &Error{err: "bad key size"}
	ErrNoSig         error = &Error{err: "no signature found"}
	ErrPrivKey       error = &Error{err: "bad private key"}
	ErrRcode         error = &Error{err: "bad rcode"}
	ErrRdata         error = &Error{err: "bad rdata"}
	ErrRRset         error = &Error{err: "bad rrset"}
	ErrSecret        error = &Error{err: "no secrets defined"}
	ErrShortRead     error = &Error{err: "short read"}
	ErrSig           error = &Error{err: "bad signature"}                      // ErrSig indicates that a signature can not be cryptographically validated.
	ErrSoa           error = &Error{err: "no SOA"}                             // ErrSOA indicates that no SOA RR was seen when doing zone transfers.
	ErrTime          error = &Error{err: "bad time"}                           // ErrTime indicates a timing error in TSIG authentication.
	ErrTruncated     error = &Error{err: "failed to unpack truncated message"} // ErrTruncated indicates that we failed to unpack a truncated message. We unpacked as much as we had so Msg can still be used, if desired.
)

// Id, by default, returns a 16 bits random number to be used as a
// message id. The random provided should be good enough. This being a
// variable the function can be reassigned to a custom function.
// For instance, to make it return a static value:
//
//	dns.Id = func() uint16 { return 3 }
var Id func() uint16 = id

// id returns a 16 bits random number to be used as a
// message id. The random provided should be good enough.
func id() uint16 {
	id32 := rand.Uint32()
	return uint16(id32)
}

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
	Compress bool       `json:"-"` // If true, the message will be compressed when converted to wire format.
	Question []Question // Holds the RR(s) of the question section.
	Answer   []RR       // Holds the RR(s) of the answer section.
	Ns       []RR       // Holds the RR(s) of the authority section.
	Extra    []RR       // Holds the RR(s) of the additional section.
}

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
	RcodeYXDomain:       "YXDOMAIN", // See RFC 2136
	RcodeYXRrset:        "YXRRSET",
	RcodeNXRrset:        "NXRRSET",
	RcodeNotAuth:        "NOTAUTH",
	RcodeNotZone:        "NOTZONE",
	RcodeBadSig:         "BADSIG", // Also known as RcodeBadVers, see RFC 6891
	//	RcodeBadVers:        "BADVERS",
	RcodeBadKey:    "BADKEY",
	RcodeBadTime:   "BADTIME",
	RcodeBadMode:   "BADMODE",
	RcodeBadName:   "BADNAME",
	RcodeBadAlg:    "BADALG",
	RcodeBadTrunc:  "BADTRUNC",
	RcodeBadCookie: "BADCOOKIE",
}

// Domain names are a sequence of counted strings
// split at the dots. They end with a zero-length string.

// PackDomainName packs a domain name s into msg[off:].
// If compression is wanted compress must be true and the compression
// map needs to hold a mapping between domain names and offsets
// pointing into msg.
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
			// Don't try to compress '.'
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
		binary.BigEndian.PutUint16(msg[nameoffset:], uint16(pointer^0xC000))
		off = nameoffset + 1
		goto End
	}
	if msg != nil && off < len(msg) {
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
	if len(s) == 0 {
		s = []byte(".")
	}
	return string(s), off1, nil
}

func packTxt(txt []string, msg []byte, offset int, tmp []byte) (int, error) {
	if len(txt) == 0 {
		if offset >= len(msg) {
			return offset, ErrBuf
		}
		msg[offset] = 0
		return offset, nil
	}
	var err error
	for i := range txt {
		if len(txt[i]) > len(tmp) {
			return offset, ErrBuf
		}
		offset, err = packTxtString(txt[i], msg, offset, tmp)
		if err != nil {
			return offset, err
		}
	}
	return offset, nil
}

func packTxtString(s string, msg []byte, offset int, tmp []byte) (int, error) {
	lenByteOffset := offset
	if offset >= len(msg) || len(s) > len(tmp) {
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

func packOctetString(s string, msg []byte, offset int, tmp []byte) (int, error) {
	if offset >= len(msg) || len(s) > len(tmp) {
		return offset, ErrBuf
	}
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
			} else {
				msg[offset] = bs[i]
			}
		} else {
			msg[offset] = bs[i]
		}
		offset++
	}
	return offset, nil
}

func unpackTxt(msg []byte, off0 int) (ss []string, off int, err error) {
	off = off0
	var s string
	for off < len(msg) && err == nil {
		s, off, err = unpackTxtString(msg, off)
		if err == nil {
			ss = append(ss, s)
		}
	}
	return
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

// Helpers for dealing with escaped bytes
func isDigit(b byte) bool { return b >= '0' && b <= '9' }

func dddToByte(s []byte) byte {
	return byte((s[0]-'0')*100 + (s[1]-'0')*10 + (s[2] - '0'))
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

// PackRR packs a resource record rr into msg[off:].
// See PackDomainName for documentation about the compression.
func PackRR(rr RR, msg []byte, off int, compression map[string]int, compress bool) (off1 int, err error) {
	if rr == nil {
		return len(msg), &Error{err: "nil rr"}
	}

	off1, err = rr.pack(msg, off, compression, compress)
	if err != nil {
		return len(msg), err
	}
	// TODO(miek): Not sure if this is needed? If removed we can remove rawmsg.go as well.
	if rawSetRdlength(msg, off, off1) {
		return off1, nil
	}
	return off, ErrRdata
}

// UnpackRR unpacks msg[off:] into an RR.
func UnpackRR(msg []byte, off int) (rr RR, off1 int, err error) {
	h, off, msg, err := unpackHeader(msg, off)
	if err != nil {
		return nil, len(msg), err
	}
	end := off + int(h.Rdlength)

	if fn, known := typeToUnpack[h.Rrtype]; !known {
		rr, off, err = unpackRFC3597(h, msg, off)
	} else {
		rr, off, err = fn(h, msg, off)
	}
	if off != end {
		return &h, end, &Error{err: "bad rdlength"}
	}
	return rr, off, err
}

// unpackRRslice unpacks msg[off:] into an []RR.
// If we cannot unpack the whole array, then it will return nil
func unpackRRslice(l int, msg []byte, off int) (dst1 []RR, off1 int, err error) {
	var r RR
	// Optimistically make dst be the length that was sent
	dst := make([]RR, 0, l)
	for i := 0; i < l; i++ {
		off1 := off
		r, off, err = UnpackRR(msg, off)
		if err != nil {
			off = len(msg)
			break
		}
		// If offset does not increase anymore, l is a lie
		if off1 == off {
			l = i
			break
		}
		dst = append(dst, r)
	}
	if err != nil && off == len(msg) {
		dst = nil
	}
	return dst, off, err
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
	// We use a similar function in tsig.go's stripTsig.
	var (
		dh          Header
		compression map[string]int
	)

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
	off, err = dh.pack(msg, off, compression, dns.Compress)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(question); i++ {
		off, err = question[i].pack(msg, off, compression, dns.Compress)
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
	var (
		dh  Header
		off int
	)
	if dh, off, err = unpackMsgHdr(msg, off); err != nil {
		return err
	}
	if off == len(msg) {
		return ErrTruncated
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

	// Optimistically use the count given to us in the header
	dns.Question = make([]Question, 0, int(dh.Qdcount))

	for i := 0; i < int(dh.Qdcount); i++ {
		off1 := off
		var q Question
		q, off, err = unpackQuestion(msg, off)
		if err != nil {
			// Even if Truncated is set, we only will set ErrTruncated if we
			// actually got the questions
			return err
		}
		if off1 == off { // Offset does not increase anymore, dh.Qdcount is a lie!
			dh.Qdcount = uint16(i)
			break
		}
		dns.Question = append(dns.Question, q)
	}

	dns.Answer, off, err = unpackRRslice(int(dh.Ancount), msg, off)
	// The header counts might have been wrong so we need to update it
	dh.Ancount = uint16(len(dns.Answer))
	if err == nil {
		dns.Ns, off, err = unpackRRslice(int(dh.Nscount), msg, off)
	}
	// The header counts might have been wrong so we need to update it
	dh.Nscount = uint16(len(dns.Ns))
	if err == nil {
		dns.Extra, off, err = unpackRRslice(int(dh.Arcount), msg, off)
	}
	// The header counts might have been wrong so we need to update it
	dh.Arcount = uint16(len(dns.Extra))

	if off != len(msg) {
		// TODO(miek) make this an error?
		// use PackOpt to let people tell how detailed the error reporting should be?
		// println("dns: extra bytes in dns packet", off, "<", len(msg))
	} else if dns.Truncated {
		// Whether we ran into a an error or not, we want to return that it
		// was truncated
		err = ErrTruncated
	}
	return err
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
		if dns.Answer[i] == nil {
			continue
		}
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
		if dns.Ns[i] == nil {
			continue
		}
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
		if dns.Extra[i] == nil {
			continue
		}
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

// TODO(miek): should add all types, because the all can be *used* for compression. Autogenerate from msg_generate and put in zmsg.go
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
	case *RP:
		compressionLenHelper(c, x.Mbox)
		compressionLenHelper(c, x.Txt)
	case *MINFO:
		compressionLenHelper(c, x.Rmail)
		compressionLenHelper(c, x.Email)
	case *AFSDB:
		compressionLenHelper(c, x.Hostname)
	case *SRV:
		compressionLenHelper(c, x.Target)
	case *NAPTR:
		compressionLenHelper(c, x.Replacement)
	case *RRSIG:
		compressionLenHelper(c, x.SignerName)
	case *NSEC:
		compressionLenHelper(c, x.NextDomain)
		// HIP?
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
	case *DNAME:
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

// Copy returns a new RR which is a deep-copy of r.
func Copy(r RR) RR { r1 := r.copy(); return r1 }

// Len returns the length (in octets) of the uncompressed RR in wire format.
func Len(r RR) int { return r.len() }

// Copy returns a new *Msg which is a deep-copy of dns.
func (dns *Msg) Copy() *Msg { return dns.CopyTo(new(Msg)) }

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

func (q *Question) pack(msg []byte, off int, compression map[string]int, compress bool) (int, error) {
	off, err := PackDomainName(q.Name, msg, off, compression, compress)
	if err != nil {
		return off, err
	}
	off, err = packUint16(q.Qtype, msg, off)
	if err != nil {
		return off, err
	}
	off, err = packUint16(q.Qclass, msg, off)
	if err != nil {
		return off, err
	}
	return off, nil
}

func unpackQuestion(msg []byte, off int) (Question, int, error) {
	var (
		q   Question
		err error
	)
	q.Name, off, err = UnpackDomainName(msg, off)
	if err != nil {
		return q, off, err
	}
	if off == len(msg) {
		return q, off, nil
	}
	q.Qtype, off, err = unpackUint16(msg, off)
	if err != nil {
		return q, off, err
	}
	if off == len(msg) {
		return q, off, nil
	}
	q.Qclass, off, err = unpackUint16(msg, off)
	if off == len(msg) {
		return q, off, nil
	}
	return q, off, err
}

func (dh *Header) pack(msg []byte, off int, compression map[string]int, compress bool) (int, error) {
	off, err := packUint16(dh.Id, msg, off)
	if err != nil {
		return off, err
	}
	off, err = packUint16(dh.Bits, msg, off)
	if err != nil {
		return off, err
	}
	off, err = packUint16(dh.Qdcount, msg, off)
	if err != nil {
		return off, err
	}
	off, err = packUint16(dh.Ancount, msg, off)
	if err != nil {
		return off, err
	}
	off, err = packUint16(dh.Nscount, msg, off)
	if err != nil {
		return off, err
	}
	off, err = packUint16(dh.Arcount, msg, off)
	return off, err
}

func unpackMsgHdr(msg []byte, off int) (Header, int, error) {
	var (
		dh  Header
		err error
	)
	dh.Id, off, err = unpackUint16(msg, off)
	if err != nil {
		return dh, off, err
	}
	dh.Bits, off, err = unpackUint16(msg, off)
	if err != nil {
		return dh, off, err
	}
	dh.Qdcount, off, err = unpackUint16(msg, off)
	if err != nil {
		return dh, off, err
	}
	dh.Ancount, off, err = unpackUint16(msg, off)
	if err != nil {
		return dh, off, err
	}
	dh.Nscount, off, err = unpackUint16(msg, off)
	if err != nil {
		return dh, off, err
	}
	dh.Arcount, off, err = unpackUint16(msg, off)
	return dh, off, err
}
