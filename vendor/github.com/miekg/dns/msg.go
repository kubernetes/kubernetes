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
	"crypto/rand"
	"encoding/binary"
	"fmt"
	"math/big"
	"strconv"
	"strings"
)

const (
	maxCompressionOffset    = 2 << 13 // We have 14 bits for the compression pointer
	maxDomainNameWireOctets = 255     // See RFC 1035 section 2.3.4

	// This is the maximum number of compression pointers that should occur in a
	// semantically valid message. Each label in a domain name must be at least one
	// octet and is separated by a period. The root label won't be represented by a
	// compression pointer to a compression pointer, hence the -2 to exclude the
	// smallest valid root label.
	//
	// It is possible to construct a valid message that has more compression pointers
	// than this, and still doesn't loop, by pointing to a previous pointer. This is
	// not something a well written implementation should ever do, so we leave them
	// to trip the maximum compression pointer check.
	maxCompressionPointers = (maxDomainNameWireOctets+1)/2 - 2

	// This is the maximum length of a domain name in presentation format. The
	// maximum wire length of a domain name is 255 octets (see above), with the
	// maximum label length being 63. The wire format requires one extra byte over
	// the presentation format, reducing the number of octets by 1. Each label in
	// the name will be separated by a single period, with each octet in the label
	// expanding to at most 4 bytes (\DDD). If all other labels are of the maximum
	// length, then the final label can only be 61 octets long to not exceed the
	// maximum allowed wire length.
	maxDomainNamePresentationLength = 61*4 + 1 + 63*4 + 1 + 63*4 + 1 + 63*4 + 1
)

// Errors defined in this package.
var (
	ErrAlg           error = &Error{err: "bad algorithm"}                  // ErrAlg indicates an error with the (DNSSEC) algorithm.
	ErrAuth          error = &Error{err: "bad authentication"}             // ErrAuth indicates an error in the TSIG authentication.
	ErrBuf           error = &Error{err: "buffer size too small"}          // ErrBuf indicates that the buffer used is too small for the message.
	ErrConnEmpty     error = &Error{err: "conn has no connection"}         // ErrConnEmpty indicates a connection is being used before it is initialized.
	ErrExtendedRcode error = &Error{err: "bad extended rcode"}             // ErrExtendedRcode ...
	ErrFqdn          error = &Error{err: "domain must be fully qualified"} // ErrFqdn indicates that a domain name does not have a closing dot.
	ErrId            error = &Error{err: "id mismatch"}                    // ErrId indicates there is a mismatch with the message's ID.
	ErrKeyAlg        error = &Error{err: "bad key algorithm"}              // ErrKeyAlg indicates that the algorithm in the key is not valid.
	ErrKey           error = &Error{err: "bad key"}
	ErrKeySize       error = &Error{err: "bad key size"}
	ErrLongDomain    error = &Error{err: fmt.Sprintf("domain name exceeded %d wire-format octets", maxDomainNameWireOctets)}
	ErrNoSig         error = &Error{err: "no signature found"}
	ErrPrivKey       error = &Error{err: "bad private key"}
	ErrRcode         error = &Error{err: "bad rcode"}
	ErrRdata         error = &Error{err: "bad rdata"}
	ErrRRset         error = &Error{err: "bad rrset"}
	ErrSecret        error = &Error{err: "no secrets defined"}
	ErrShortRead     error = &Error{err: "short read"}
	ErrSig           error = &Error{err: "bad signature"} // ErrSig indicates that a signature can not be cryptographically validated.
	ErrSoa           error = &Error{err: "no SOA"}        // ErrSOA indicates that no SOA RR was seen when doing zone transfers.
	ErrTime          error = &Error{err: "bad time"}      // ErrTime indicates a timing error in TSIG authentication.
)

// Id by default returns a 16-bit random number to be used as a message id. The
// number is drawn from a cryptographically secure random number generator.
// This being a variable the function can be reassigned to a custom function.
// For instance, to make it return a static value for testing:
//
//	dns.Id = func() uint16 { return 3 }
var Id = id

// id returns a 16 bits random number to be used as a
// message id. The random provided should be good enough.
func id() uint16 {
	var output uint16
	err := binary.Read(rand.Reader, binary.BigEndian, &output)
	if err != nil {
		panic("dns: reading random id failed: " + err.Error())
	}
	return output
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
	RcodeNotImplemented: "NOTIMP",
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

// compressionMap is used to allow a more efficient compression map
// to be used for internal packDomainName calls without changing the
// signature or functionality of public API.
//
// In particular, map[string]uint16 uses 25% less per-entry memory
// than does map[string]int.
type compressionMap struct {
	ext map[string]int    // external callers
	int map[string]uint16 // internal callers
}

func (m compressionMap) valid() bool {
	return m.int != nil || m.ext != nil
}

func (m compressionMap) insert(s string, pos int) {
	if m.ext != nil {
		m.ext[s] = pos
	} else {
		m.int[s] = uint16(pos)
	}
}

func (m compressionMap) find(s string) (int, bool) {
	if m.ext != nil {
		pos, ok := m.ext[s]
		return pos, ok
	}

	pos, ok := m.int[s]
	return int(pos), ok
}

// Domain names are a sequence of counted strings
// split at the dots. They end with a zero-length string.

// PackDomainName packs a domain name s into msg[off:].
// If compression is wanted compress must be true and the compression
// map needs to hold a mapping between domain names and offsets
// pointing into msg.
func PackDomainName(s string, msg []byte, off int, compression map[string]int, compress bool) (off1 int, err error) {
	return packDomainName(s, msg, off, compressionMap{ext: compression}, compress)
}

func packDomainName(s string, msg []byte, off int, compression compressionMap, compress bool) (off1 int, err error) {
	// XXX: A logical copy of this function exists in IsDomainName and
	// should be kept in sync with this function.

	ls := len(s)
	if ls == 0 { // Ok, for instance when dealing with update RR without any rdata.
		return off, nil
	}

	// If not fully qualified, error out.
	if !IsFqdn(s) {
		return len(msg), ErrFqdn
	}

	// Each dot ends a segment of the name.
	// We trade each dot byte for a length byte.
	// Except for escaped dots (\.), which are normal dots.
	// There is also a trailing zero.

	// Compression
	pointer := -1

	// Emit sequence of counted strings, chopping at dots.
	var (
		begin     int
		compBegin int
		compOff   int
		bs        []byte
		wasDot    bool
	)
loop:
	for i := 0; i < ls; i++ {
		var c byte
		if bs == nil {
			c = s[i]
		} else {
			c = bs[i]
		}

		switch c {
		case '\\':
			if off+1 > len(msg) {
				return len(msg), ErrBuf
			}

			if bs == nil {
				bs = []byte(s)
			}

			// check for \DDD
			if i+3 < ls && isDigit(bs[i+1]) && isDigit(bs[i+2]) && isDigit(bs[i+3]) {
				bs[i] = dddToByte(bs[i+1:])
				copy(bs[i+1:ls-3], bs[i+4:])
				ls -= 3
				compOff += 3
			} else {
				copy(bs[i:ls-1], bs[i+1:])
				ls--
				compOff++
			}

			wasDot = false
		case '.':
			if wasDot {
				// two dots back to back is not legal
				return len(msg), ErrRdata
			}
			wasDot = true

			labelLen := i - begin
			if labelLen >= 1<<6 { // top two bits of length must be clear
				return len(msg), ErrRdata
			}

			// off can already (we're in a loop) be bigger than len(msg)
			// this happens when a name isn't fully qualified
			if off+1+labelLen > len(msg) {
				return len(msg), ErrBuf
			}

			// Don't try to compress '.'
			// We should only compress when compress is true, but we should also still pick
			// up names that can be used for *future* compression(s).
			if compression.valid() && !isRootLabel(s, bs, begin, ls) {
				if p, ok := compression.find(s[compBegin:]); ok {
					// The first hit is the longest matching dname
					// keep the pointer offset we get back and store
					// the offset of the current name, because that's
					// where we need to insert the pointer later

					// If compress is true, we're allowed to compress this dname
					if compress {
						pointer = p // Where to point to
						break loop
					}
				} else if off < maxCompressionOffset {
					// Only offsets smaller than maxCompressionOffset can be used.
					compression.insert(s[compBegin:], off)
				}
			}

			// The following is covered by the length check above.
			msg[off] = byte(labelLen)

			if bs == nil {
				copy(msg[off+1:], s[begin:i])
			} else {
				copy(msg[off+1:], bs[begin:i])
			}
			off += 1 + labelLen

			begin = i + 1
			compBegin = begin + compOff
		default:
			wasDot = false
		}
	}

	// Root label is special
	if isRootLabel(s, bs, 0, ls) {
		return off, nil
	}

	// If we did compression and we find something add the pointer here
	if pointer != -1 {
		// We have two bytes (14 bits) to put the pointer in
		binary.BigEndian.PutUint16(msg[off:], uint16(pointer^0xC000))
		return off + 2, nil
	}

	if off < len(msg) {
		msg[off] = 0
	}

	return off + 1, nil
}

// isRootLabel returns whether s or bs, from off to end, is the root
// label ".".
//
// If bs is nil, s will be checked, otherwise bs will be checked.
func isRootLabel(s string, bs []byte, off, end int) bool {
	if bs == nil {
		return s[off:end] == "."
	}

	return end-off == 1 && bs[off] == '.'
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

// UnpackDomainName unpacks a domain name into a string. It returns
// the name, the new offset into msg and any error that occurred.
//
// When an error is encountered, the unpacked name will be discarded
// and len(msg) will be returned as the offset.
func UnpackDomainName(msg []byte, off int) (string, int, error) {
	s := make([]byte, 0, maxDomainNamePresentationLength)
	off1 := 0
	lenmsg := len(msg)
	budget := maxDomainNameWireOctets
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
			budget -= c + 1 // +1 for the label separator
			if budget <= 0 {
				return "", lenmsg, ErrLongDomain
			}
			for _, b := range msg[off : off+c] {
				if isDomainNameLabelSpecial(b) {
					s = append(s, '\\', b)
				} else if b < ' ' || b > '~' {
					s = append(s, escapeByte(b)...)
				} else {
					s = append(s, b)
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
			if ptr++; ptr > maxCompressionPointers {
				return "", lenmsg, &Error{err: "too many compression pointers"}
			}
			// pointer should guarantee that it advances and points forwards at least
			// but the condition on previous three lines guarantees that it's
			// at least loop-free
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
		return ".", off1, nil
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
	for _, s := range txt {
		if len(s) > len(tmp) {
			return offset, ErrBuf
		}
		offset, err = packTxtString(s, msg, offset, tmp)
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
		s, off, err = unpackString(msg, off)
		if err == nil {
			ss = append(ss, s)
		}
	}
	return
}

// Helpers for dealing with escaped bytes
func isDigit(b byte) bool { return b >= '0' && b <= '9' }

func dddToByte(s []byte) byte {
	_ = s[2] // bounds check hint to compiler; see golang.org/issue/14808
	return byte((s[0]-'0')*100 + (s[1]-'0')*10 + (s[2] - '0'))
}

func dddStringToByte(s string) byte {
	_ = s[2] // bounds check hint to compiler; see golang.org/issue/14808
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
	headerEnd, off1, err := packRR(rr, msg, off, compressionMap{ext: compression}, compress)
	if err == nil {
		// packRR no longer sets the Rdlength field on the rr, but
		// callers might be expecting it so we set it here.
		rr.Header().Rdlength = uint16(off1 - headerEnd)
	}
	return off1, err
}

func packRR(rr RR, msg []byte, off int, compression compressionMap, compress bool) (headerEnd int, off1 int, err error) {
	if rr == nil {
		return len(msg), len(msg), &Error{err: "nil rr"}
	}

	headerEnd, err = rr.Header().packHeader(msg, off, compression, compress)
	if err != nil {
		return headerEnd, len(msg), err
	}

	off1, err = rr.pack(msg, headerEnd, compression, compress)
	if err != nil {
		return headerEnd, len(msg), err
	}

	rdlength := off1 - headerEnd
	if int(uint16(rdlength)) != rdlength { // overflow
		return headerEnd, len(msg), ErrRdata
	}

	// The RDLENGTH field is the last field in the header and we set it here.
	binary.BigEndian.PutUint16(msg[headerEnd-2:], uint16(rdlength))
	return headerEnd, off1, nil
}

// UnpackRR unpacks msg[off:] into an RR.
func UnpackRR(msg []byte, off int) (rr RR, off1 int, err error) {
	h, off, msg, err := unpackHeader(msg, off)
	if err != nil {
		return nil, len(msg), err
	}

	return UnpackRRWithHeader(h, msg, off)
}

// UnpackRRWithHeader unpacks the record type specific payload given an existing
// RR_Header.
func UnpackRRWithHeader(h RR_Header, msg []byte, off int) (rr RR, off1 int, err error) {
	if newFn, ok := TypeToRR[h.Rrtype]; ok {
		rr = newFn()
		*rr.Header() = h
	} else {
		rr = &RFC3597{Hdr: h}
	}

	if noRdata(h) {
		return rr, off, nil
	}

	end := off + int(h.Rdlength)

	off, err = rr.unpack(msg, off)
	if err != nil {
		return nil, end, err
	}
	if off != end {
		return &h, end, &Error{err: "bad rdlength"}
	}

	return rr, off, nil
}

// unpackRRslice unpacks msg[off:] into an []RR.
// If we cannot unpack the whole array, then it will return nil
func unpackRRslice(l int, msg []byte, off int) (dst1 []RR, off1 int, err error) {
	var r RR
	// Don't pre-allocate, l may be under attacker control
	var dst []RR
	for i := 0; i < l; i++ {
		off1 := off
		r, off, err = UnpackRR(msg, off)
		if err != nil {
			off = len(msg)
			break
		}
		// If offset does not increase anymore, l is a lie
		if off1 == off {
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

// PackBuffer packs a Msg, using the given buffer buf. If buf is too small a new buffer is allocated.
func (dns *Msg) PackBuffer(buf []byte) (msg []byte, err error) {
	// If this message can't be compressed, avoid filling the
	// compression map and creating garbage.
	if dns.Compress && dns.isCompressible() {
		compression := make(map[string]uint16) // Compression pointer mappings.
		return dns.packBufferWithCompressionMap(buf, compressionMap{int: compression}, true)
	}

	return dns.packBufferWithCompressionMap(buf, compressionMap{}, false)
}

// packBufferWithCompressionMap packs a Msg, using the given buffer buf.
func (dns *Msg) packBufferWithCompressionMap(buf []byte, compression compressionMap, compress bool) (msg []byte, err error) {
	if dns.Rcode < 0 || dns.Rcode > 0xFFF {
		return nil, ErrRcode
	}

	// Set extended rcode unconditionally if we have an opt, this will allow
	// reseting the extended rcode bits if they need to.
	if opt := dns.IsEdns0(); opt != nil {
		opt.SetExtendedRcode(uint16(dns.Rcode))
	} else if dns.Rcode > 0xF {
		// If Rcode is an extended one and opt is nil, error out.
		return nil, ErrExtendedRcode
	}

	// Convert convenient Msg into wire-like Header.
	var dh Header
	dh.Id = dns.Id
	dh.Bits = uint16(dns.Opcode)<<11 | uint16(dns.Rcode&0xF)
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

	dh.Qdcount = uint16(len(dns.Question))
	dh.Ancount = uint16(len(dns.Answer))
	dh.Nscount = uint16(len(dns.Ns))
	dh.Arcount = uint16(len(dns.Extra))

	// We need the uncompressed length here, because we first pack it and then compress it.
	msg = buf
	uncompressedLen := msgLenWithCompressionMap(dns, nil)
	if packLen := uncompressedLen + 1; len(msg) < packLen {
		msg = make([]byte, packLen)
	}

	// Pack it in: header and then the pieces.
	off := 0
	off, err = dh.pack(msg, off, compression, compress)
	if err != nil {
		return nil, err
	}
	for _, r := range dns.Question {
		off, err = r.pack(msg, off, compression, compress)
		if err != nil {
			return nil, err
		}
	}
	for _, r := range dns.Answer {
		_, off, err = packRR(r, msg, off, compression, compress)
		if err != nil {
			return nil, err
		}
	}
	for _, r := range dns.Ns {
		_, off, err = packRR(r, msg, off, compression, compress)
		if err != nil {
			return nil, err
		}
	}
	for _, r := range dns.Extra {
		_, off, err = packRR(r, msg, off, compression, compress)
		if err != nil {
			return nil, err
		}
	}
	return msg[:off], nil
}

func (dns *Msg) unpack(dh Header, msg []byte, off int) (err error) {
	// If we are at the end of the message we should return *just* the
	// header. This can still be useful to the caller. 9.9.9.9 sends these
	// when responding with REFUSED for instance.
	if off == len(msg) {
		// reset sections before returning
		dns.Question, dns.Answer, dns.Ns, dns.Extra = nil, nil, nil, nil
		return nil
	}

	// Qdcount, Ancount, Nscount, Arcount can't be trusted, as they are
	// attacker controlled. This means we can't use them to pre-allocate
	// slices.
	dns.Question = nil
	for i := 0; i < int(dh.Qdcount); i++ {
		off1 := off
		var q Question
		q, off, err = unpackQuestion(msg, off)
		if err != nil {
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

	// Set extended Rcode
	if opt := dns.IsEdns0(); opt != nil {
		dns.Rcode |= opt.ExtendedRcode()
	}

	if off != len(msg) {
		// TODO(miek) make this an error?
		// use PackOpt to let people tell how detailed the error reporting should be?
		// println("dns: extra bytes in dns packet", off, "<", len(msg))
	}
	return err

}

// Unpack unpacks a binary message to a Msg structure.
func (dns *Msg) Unpack(msg []byte) (err error) {
	dh, off, err := unpackMsgHdr(msg, 0)
	if err != nil {
		return err
	}

	dns.setHdr(dh)
	return dns.unpack(dh, msg, off)
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
		for _, r := range dns.Question {
			s += r.String() + "\n"
		}
	}
	if len(dns.Answer) > 0 {
		s += "\n;; ANSWER SECTION:\n"
		for _, r := range dns.Answer {
			if r != nil {
				s += r.String() + "\n"
			}
		}
	}
	if len(dns.Ns) > 0 {
		s += "\n;; AUTHORITY SECTION:\n"
		for _, r := range dns.Ns {
			if r != nil {
				s += r.String() + "\n"
			}
		}
	}
	if len(dns.Extra) > 0 {
		s += "\n;; ADDITIONAL SECTION:\n"
		for _, r := range dns.Extra {
			if r != nil {
				s += r.String() + "\n"
			}
		}
	}
	return s
}

// isCompressible returns whether the msg may be compressible.
func (dns *Msg) isCompressible() bool {
	// If we only have one question, there is nothing we can ever compress.
	return len(dns.Question) > 1 || len(dns.Answer) > 0 ||
		len(dns.Ns) > 0 || len(dns.Extra) > 0
}

// Len returns the message length when in (un)compressed wire format.
// If dns.Compress is true compression it is taken into account. Len()
// is provided to be a faster way to get the size of the resulting packet,
// than packing it, measuring the size and discarding the buffer.
func (dns *Msg) Len() int {
	// If this message can't be compressed, avoid filling the
	// compression map and creating garbage.
	if dns.Compress && dns.isCompressible() {
		compression := make(map[string]struct{})
		return msgLenWithCompressionMap(dns, compression)
	}

	return msgLenWithCompressionMap(dns, nil)
}

func msgLenWithCompressionMap(dns *Msg, compression map[string]struct{}) int {
	l := headerSize

	for _, r := range dns.Question {
		l += r.len(l, compression)
	}
	for _, r := range dns.Answer {
		if r != nil {
			l += r.len(l, compression)
		}
	}
	for _, r := range dns.Ns {
		if r != nil {
			l += r.len(l, compression)
		}
	}
	for _, r := range dns.Extra {
		if r != nil {
			l += r.len(l, compression)
		}
	}

	return l
}

func domainNameLen(s string, off int, compression map[string]struct{}, compress bool) int {
	if s == "" || s == "." {
		return 1
	}

	escaped := strings.Contains(s, "\\")

	if compression != nil && (compress || off < maxCompressionOffset) {
		// compressionLenSearch will insert the entry into the compression
		// map if it doesn't contain it.
		if l, ok := compressionLenSearch(compression, s, off); ok && compress {
			if escaped {
				return escapedNameLen(s[:l]) + 2
			}

			return l + 2
		}
	}

	if escaped {
		return escapedNameLen(s) + 1
	}

	return len(s) + 1
}

func escapedNameLen(s string) int {
	nameLen := len(s)
	for i := 0; i < len(s); i++ {
		if s[i] != '\\' {
			continue
		}

		if i+3 < len(s) && isDigit(s[i+1]) && isDigit(s[i+2]) && isDigit(s[i+3]) {
			nameLen -= 3
			i += 3
		} else {
			nameLen--
			i++
		}
	}

	return nameLen
}

func compressionLenSearch(c map[string]struct{}, s string, msgOff int) (int, bool) {
	for off, end := 0, false; !end; off, end = NextLabel(s, off) {
		if _, ok := c[s[off:]]; ok {
			return off, true
		}

		if msgOff+off < maxCompressionOffset {
			c[s[off:]] = struct{}{}
		}
	}

	return 0, false
}

// Copy returns a new RR which is a deep-copy of r.
func Copy(r RR) RR { return r.copy() }

// Len returns the length (in octets) of the uncompressed RR in wire format.
func Len(r RR) int { return r.len(0, nil) }

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
	r1.Answer, rrArr = rrArr[:0:len(dns.Answer)], rrArr[len(dns.Answer):]
	r1.Ns, rrArr = rrArr[:0:len(dns.Ns)], rrArr[len(dns.Ns):]
	r1.Extra = rrArr[:0:len(dns.Extra)]

	for _, r := range dns.Answer {
		r1.Answer = append(r1.Answer, r.copy())
	}

	for _, r := range dns.Ns {
		r1.Ns = append(r1.Ns, r.copy())
	}

	for _, r := range dns.Extra {
		r1.Extra = append(r1.Extra, r.copy())
	}

	return r1
}

func (q *Question) pack(msg []byte, off int, compression compressionMap, compress bool) (int, error) {
	off, err := packDomainName(q.Name, msg, off, compression, compress)
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

func (dh *Header) pack(msg []byte, off int, compression compressionMap, compress bool) (int, error) {
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
	if err != nil {
		return off, err
	}
	return off, nil
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
	if err != nil {
		return dh, off, err
	}
	return dh, off, nil
}

// setHdr set the header in the dns using the binary data in dh.
func (dns *Msg) setHdr(dh Header) {
	dns.Id = dh.Id
	dns.Response = dh.Bits&_QR != 0
	dns.Opcode = int(dh.Bits>>11) & 0xF
	dns.Authoritative = dh.Bits&_AA != 0
	dns.Truncated = dh.Bits&_TC != 0
	dns.RecursionDesired = dh.Bits&_RD != 0
	dns.RecursionAvailable = dh.Bits&_RA != 0
	dns.Zero = dh.Bits&_Z != 0 // _Z covers the zero bit, which should be zero; not sure why we set it to the opposite.
	dns.AuthenticatedData = dh.Bits&_AD != 0
	dns.CheckingDisabled = dh.Bits&_CD != 0
	dns.Rcode = int(dh.Bits & 0xF)
}
