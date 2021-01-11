package dns

import (
	"encoding/base32"
	"encoding/base64"
	"encoding/binary"
	"encoding/hex"
	"net"
	"sort"
	"strings"
)

// helper functions called from the generated zmsg.go

// These function are named after the tag to help pack/unpack, if there is no tag it is the name
// of the type they pack/unpack (string, int, etc). We prefix all with unpackData or packData, so packDataA or
// packDataDomainName.

func unpackDataA(msg []byte, off int) (net.IP, int, error) {
	if off+net.IPv4len > len(msg) {
		return nil, len(msg), &Error{err: "overflow unpacking a"}
	}
	a := append(make(net.IP, 0, net.IPv4len), msg[off:off+net.IPv4len]...)
	off += net.IPv4len
	return a, off, nil
}

func packDataA(a net.IP, msg []byte, off int) (int, error) {
	switch len(a) {
	case net.IPv4len, net.IPv6len:
		// It must be a slice of 4, even if it is 16, we encode only the first 4
		if off+net.IPv4len > len(msg) {
			return len(msg), &Error{err: "overflow packing a"}
		}

		copy(msg[off:], a.To4())
		off += net.IPv4len
	case 0:
		// Allowed, for dynamic updates.
	default:
		return len(msg), &Error{err: "overflow packing a"}
	}
	return off, nil
}

func unpackDataAAAA(msg []byte, off int) (net.IP, int, error) {
	if off+net.IPv6len > len(msg) {
		return nil, len(msg), &Error{err: "overflow unpacking aaaa"}
	}
	aaaa := append(make(net.IP, 0, net.IPv6len), msg[off:off+net.IPv6len]...)
	off += net.IPv6len
	return aaaa, off, nil
}

func packDataAAAA(aaaa net.IP, msg []byte, off int) (int, error) {
	switch len(aaaa) {
	case net.IPv6len:
		if off+net.IPv6len > len(msg) {
			return len(msg), &Error{err: "overflow packing aaaa"}
		}

		copy(msg[off:], aaaa)
		off += net.IPv6len
	case 0:
		// Allowed, dynamic updates.
	default:
		return len(msg), &Error{err: "overflow packing aaaa"}
	}
	return off, nil
}

// unpackHeader unpacks an RR header, returning the offset to the end of the header and a
// re-sliced msg according to the expected length of the RR.
func unpackHeader(msg []byte, off int) (rr RR_Header, off1 int, truncmsg []byte, err error) {
	hdr := RR_Header{}
	if off == len(msg) {
		return hdr, off, msg, nil
	}

	hdr.Name, off, err = UnpackDomainName(msg, off)
	if err != nil {
		return hdr, len(msg), msg, err
	}
	hdr.Rrtype, off, err = unpackUint16(msg, off)
	if err != nil {
		return hdr, len(msg), msg, err
	}
	hdr.Class, off, err = unpackUint16(msg, off)
	if err != nil {
		return hdr, len(msg), msg, err
	}
	hdr.Ttl, off, err = unpackUint32(msg, off)
	if err != nil {
		return hdr, len(msg), msg, err
	}
	hdr.Rdlength, off, err = unpackUint16(msg, off)
	if err != nil {
		return hdr, len(msg), msg, err
	}
	msg, err = truncateMsgFromRdlength(msg, off, hdr.Rdlength)
	return hdr, off, msg, err
}

// packHeader packs an RR header, returning the offset to the end of the header.
// See PackDomainName for documentation about the compression.
func (hdr RR_Header) packHeader(msg []byte, off int, compression compressionMap, compress bool) (int, error) {
	if off == len(msg) {
		return off, nil
	}

	off, err := packDomainName(hdr.Name, msg, off, compression, compress)
	if err != nil {
		return len(msg), err
	}
	off, err = packUint16(hdr.Rrtype, msg, off)
	if err != nil {
		return len(msg), err
	}
	off, err = packUint16(hdr.Class, msg, off)
	if err != nil {
		return len(msg), err
	}
	off, err = packUint32(hdr.Ttl, msg, off)
	if err != nil {
		return len(msg), err
	}
	off, err = packUint16(0, msg, off) // The RDLENGTH field will be set later in packRR.
	if err != nil {
		return len(msg), err
	}
	return off, nil
}

// helper helper functions.

// truncateMsgFromRdLength truncates msg to match the expected length of the RR.
// Returns an error if msg is smaller than the expected size.
func truncateMsgFromRdlength(msg []byte, off int, rdlength uint16) (truncmsg []byte, err error) {
	lenrd := off + int(rdlength)
	if lenrd > len(msg) {
		return msg, &Error{err: "overflowing header size"}
	}
	return msg[:lenrd], nil
}

var base32HexNoPadEncoding = base32.HexEncoding.WithPadding(base32.NoPadding)

func fromBase32(s []byte) (buf []byte, err error) {
	for i, b := range s {
		if b >= 'a' && b <= 'z' {
			s[i] = b - 32
		}
	}
	buflen := base32HexNoPadEncoding.DecodedLen(len(s))
	buf = make([]byte, buflen)
	n, err := base32HexNoPadEncoding.Decode(buf, s)
	buf = buf[:n]
	return
}

func toBase32(b []byte) string {
	return base32HexNoPadEncoding.EncodeToString(b)
}

func fromBase64(s []byte) (buf []byte, err error) {
	buflen := base64.StdEncoding.DecodedLen(len(s))
	buf = make([]byte, buflen)
	n, err := base64.StdEncoding.Decode(buf, s)
	buf = buf[:n]
	return
}

func toBase64(b []byte) string { return base64.StdEncoding.EncodeToString(b) }

// dynamicUpdate returns true if the Rdlength is zero.
func noRdata(h RR_Header) bool { return h.Rdlength == 0 }

func unpackUint8(msg []byte, off int) (i uint8, off1 int, err error) {
	if off+1 > len(msg) {
		return 0, len(msg), &Error{err: "overflow unpacking uint8"}
	}
	return msg[off], off + 1, nil
}

func packUint8(i uint8, msg []byte, off int) (off1 int, err error) {
	if off+1 > len(msg) {
		return len(msg), &Error{err: "overflow packing uint8"}
	}
	msg[off] = i
	return off + 1, nil
}

func unpackUint16(msg []byte, off int) (i uint16, off1 int, err error) {
	if off+2 > len(msg) {
		return 0, len(msg), &Error{err: "overflow unpacking uint16"}
	}
	return binary.BigEndian.Uint16(msg[off:]), off + 2, nil
}

func packUint16(i uint16, msg []byte, off int) (off1 int, err error) {
	if off+2 > len(msg) {
		return len(msg), &Error{err: "overflow packing uint16"}
	}
	binary.BigEndian.PutUint16(msg[off:], i)
	return off + 2, nil
}

func unpackUint32(msg []byte, off int) (i uint32, off1 int, err error) {
	if off+4 > len(msg) {
		return 0, len(msg), &Error{err: "overflow unpacking uint32"}
	}
	return binary.BigEndian.Uint32(msg[off:]), off + 4, nil
}

func packUint32(i uint32, msg []byte, off int) (off1 int, err error) {
	if off+4 > len(msg) {
		return len(msg), &Error{err: "overflow packing uint32"}
	}
	binary.BigEndian.PutUint32(msg[off:], i)
	return off + 4, nil
}

func unpackUint48(msg []byte, off int) (i uint64, off1 int, err error) {
	if off+6 > len(msg) {
		return 0, len(msg), &Error{err: "overflow unpacking uint64 as uint48"}
	}
	// Used in TSIG where the last 48 bits are occupied, so for now, assume a uint48 (6 bytes)
	i = uint64(msg[off])<<40 | uint64(msg[off+1])<<32 | uint64(msg[off+2])<<24 | uint64(msg[off+3])<<16 |
		uint64(msg[off+4])<<8 | uint64(msg[off+5])
	off += 6
	return i, off, nil
}

func packUint48(i uint64, msg []byte, off int) (off1 int, err error) {
	if off+6 > len(msg) {
		return len(msg), &Error{err: "overflow packing uint64 as uint48"}
	}
	msg[off] = byte(i >> 40)
	msg[off+1] = byte(i >> 32)
	msg[off+2] = byte(i >> 24)
	msg[off+3] = byte(i >> 16)
	msg[off+4] = byte(i >> 8)
	msg[off+5] = byte(i)
	off += 6
	return off, nil
}

func unpackUint64(msg []byte, off int) (i uint64, off1 int, err error) {
	if off+8 > len(msg) {
		return 0, len(msg), &Error{err: "overflow unpacking uint64"}
	}
	return binary.BigEndian.Uint64(msg[off:]), off + 8, nil
}

func packUint64(i uint64, msg []byte, off int) (off1 int, err error) {
	if off+8 > len(msg) {
		return len(msg), &Error{err: "overflow packing uint64"}
	}
	binary.BigEndian.PutUint64(msg[off:], i)
	off += 8
	return off, nil
}

func unpackString(msg []byte, off int) (string, int, error) {
	if off+1 > len(msg) {
		return "", off, &Error{err: "overflow unpacking txt"}
	}
	l := int(msg[off])
	off++
	if off+l > len(msg) {
		return "", off, &Error{err: "overflow unpacking txt"}
	}
	var s strings.Builder
	consumed := 0
	for i, b := range msg[off : off+l] {
		switch {
		case b == '"' || b == '\\':
			if consumed == 0 {
				s.Grow(l * 2)
			}
			s.Write(msg[off+consumed : off+i])
			s.WriteByte('\\')
			s.WriteByte(b)
			consumed = i + 1
		case b < ' ' || b > '~': // unprintable
			if consumed == 0 {
				s.Grow(l * 2)
			}
			s.Write(msg[off+consumed : off+i])
			s.WriteString(escapeByte(b))
			consumed = i + 1
		}
	}
	if consumed == 0 { // no escaping needed
		return string(msg[off : off+l]), off + l, nil
	}
	s.Write(msg[off+consumed : off+l])
	return s.String(), off + l, nil
}

func packString(s string, msg []byte, off int) (int, error) {
	txtTmp := make([]byte, 256*4+1)
	off, err := packTxtString(s, msg, off, txtTmp)
	if err != nil {
		return len(msg), err
	}
	return off, nil
}

func unpackStringBase32(msg []byte, off, end int) (string, int, error) {
	if end > len(msg) {
		return "", len(msg), &Error{err: "overflow unpacking base32"}
	}
	s := toBase32(msg[off:end])
	return s, end, nil
}

func packStringBase32(s string, msg []byte, off int) (int, error) {
	b32, err := fromBase32([]byte(s))
	if err != nil {
		return len(msg), err
	}
	if off+len(b32) > len(msg) {
		return len(msg), &Error{err: "overflow packing base32"}
	}
	copy(msg[off:off+len(b32)], b32)
	off += len(b32)
	return off, nil
}

func unpackStringBase64(msg []byte, off, end int) (string, int, error) {
	// Rest of the RR is base64 encoded value, so we don't need an explicit length
	// to be set. Thus far all RR's that have base64 encoded fields have those as their
	// last one. What we do need is the end of the RR!
	if end > len(msg) {
		return "", len(msg), &Error{err: "overflow unpacking base64"}
	}
	s := toBase64(msg[off:end])
	return s, end, nil
}

func packStringBase64(s string, msg []byte, off int) (int, error) {
	b64, err := fromBase64([]byte(s))
	if err != nil {
		return len(msg), err
	}
	if off+len(b64) > len(msg) {
		return len(msg), &Error{err: "overflow packing base64"}
	}
	copy(msg[off:off+len(b64)], b64)
	off += len(b64)
	return off, nil
}

func unpackStringHex(msg []byte, off, end int) (string, int, error) {
	// Rest of the RR is hex encoded value, so we don't need an explicit length
	// to be set. NSEC and TSIG have hex fields with a length field.
	// What we do need is the end of the RR!
	if end > len(msg) {
		return "", len(msg), &Error{err: "overflow unpacking hex"}
	}

	s := hex.EncodeToString(msg[off:end])
	return s, end, nil
}

func packStringHex(s string, msg []byte, off int) (int, error) {
	h, err := hex.DecodeString(s)
	if err != nil {
		return len(msg), err
	}
	if off+len(h) > len(msg) {
		return len(msg), &Error{err: "overflow packing hex"}
	}
	copy(msg[off:off+len(h)], h)
	off += len(h)
	return off, nil
}

func unpackStringAny(msg []byte, off, end int) (string, int, error) {
	if end > len(msg) {
		return "", len(msg), &Error{err: "overflow unpacking anything"}
	}
	return string(msg[off:end]), end, nil
}

func packStringAny(s string, msg []byte, off int) (int, error) {
	if off+len(s) > len(msg) {
		return len(msg), &Error{err: "overflow packing anything"}
	}
	copy(msg[off:off+len(s)], s)
	off += len(s)
	return off, nil
}

func unpackStringTxt(msg []byte, off int) ([]string, int, error) {
	txt, off, err := unpackTxt(msg, off)
	if err != nil {
		return nil, len(msg), err
	}
	return txt, off, nil
}

func packStringTxt(s []string, msg []byte, off int) (int, error) {
	txtTmp := make([]byte, 256*4+1) // If the whole string consists out of \DDD we need this many.
	off, err := packTxt(s, msg, off, txtTmp)
	if err != nil {
		return len(msg), err
	}
	return off, nil
}

func unpackDataOpt(msg []byte, off int) ([]EDNS0, int, error) {
	var edns []EDNS0
Option:
	var code uint16
	if off+4 > len(msg) {
		return nil, len(msg), &Error{err: "overflow unpacking opt"}
	}
	code = binary.BigEndian.Uint16(msg[off:])
	off += 2
	optlen := binary.BigEndian.Uint16(msg[off:])
	off += 2
	if off+int(optlen) > len(msg) {
		return nil, len(msg), &Error{err: "overflow unpacking opt"}
	}
	e := makeDataOpt(code)
	if err := e.unpack(msg[off : off+int(optlen)]); err != nil {
		return nil, len(msg), err
	}
	edns = append(edns, e)
	off += int(optlen)

	if off < len(msg) {
		goto Option
	}

	return edns, off, nil
}

func makeDataOpt(code uint16) EDNS0 {
	switch code {
	case EDNS0NSID:
		return new(EDNS0_NSID)
	case EDNS0SUBNET:
		return new(EDNS0_SUBNET)
	case EDNS0COOKIE:
		return new(EDNS0_COOKIE)
	case EDNS0EXPIRE:
		return new(EDNS0_EXPIRE)
	case EDNS0UL:
		return new(EDNS0_UL)
	case EDNS0LLQ:
		return new(EDNS0_LLQ)
	case EDNS0DAU:
		return new(EDNS0_DAU)
	case EDNS0DHU:
		return new(EDNS0_DHU)
	case EDNS0N3U:
		return new(EDNS0_N3U)
	case EDNS0PADDING:
		return new(EDNS0_PADDING)
	default:
		e := new(EDNS0_LOCAL)
		e.Code = code
		return e
	}
}

func packDataOpt(options []EDNS0, msg []byte, off int) (int, error) {
	for _, el := range options {
		b, err := el.pack()
		if err != nil || off+4 > len(msg) {
			return len(msg), &Error{err: "overflow packing opt"}
		}
		binary.BigEndian.PutUint16(msg[off:], el.Option())      // Option code
		binary.BigEndian.PutUint16(msg[off+2:], uint16(len(b))) // Length
		off += 4
		if off+len(b) > len(msg) {
			return len(msg), &Error{err: "overflow packing opt"}
		}
		// Actual data
		copy(msg[off:off+len(b)], b)
		off += len(b)
	}
	return off, nil
}

func unpackStringOctet(msg []byte, off int) (string, int, error) {
	s := string(msg[off:])
	return s, len(msg), nil
}

func packStringOctet(s string, msg []byte, off int) (int, error) {
	txtTmp := make([]byte, 256*4+1)
	off, err := packOctetString(s, msg, off, txtTmp)
	if err != nil {
		return len(msg), err
	}
	return off, nil
}

func unpackDataNsec(msg []byte, off int) ([]uint16, int, error) {
	var nsec []uint16
	length, window, lastwindow := 0, 0, -1
	for off < len(msg) {
		if off+2 > len(msg) {
			return nsec, len(msg), &Error{err: "overflow unpacking nsecx"}
		}
		window = int(msg[off])
		length = int(msg[off+1])
		off += 2
		if window <= lastwindow {
			// RFC 4034: Blocks are present in the NSEC RR RDATA in
			// increasing numerical order.
			return nsec, len(msg), &Error{err: "out of order NSEC block"}
		}
		if length == 0 {
			// RFC 4034: Blocks with no types present MUST NOT be included.
			return nsec, len(msg), &Error{err: "empty NSEC block"}
		}
		if length > 32 {
			return nsec, len(msg), &Error{err: "NSEC block too long"}
		}
		if off+length > len(msg) {
			return nsec, len(msg), &Error{err: "overflowing NSEC block"}
		}

		// Walk the bytes in the window and extract the type bits
		for j, b := range msg[off : off+length] {
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
		lastwindow = window
	}
	return nsec, off, nil
}

// typeBitMapLen is a helper function which computes the "maximum" length of
// a the NSEC Type BitMap field.
func typeBitMapLen(bitmap []uint16) int {
	var l int
	var lastwindow, lastlength uint16
	for _, t := range bitmap {
		window := t / 256
		length := (t-window*256)/8 + 1
		if window > lastwindow && lastlength != 0 { // New window, jump to the new offset
			l += int(lastlength) + 2
			lastlength = 0
		}
		if window < lastwindow || length < lastlength {
			// packDataNsec would return Error{err: "nsec bits out of order"} here, but
			// when computing the length, we want do be liberal.
			continue
		}
		lastwindow, lastlength = window, length
	}
	l += int(lastlength) + 2
	return l
}

func packDataNsec(bitmap []uint16, msg []byte, off int) (int, error) {
	if len(bitmap) == 0 {
		return off, nil
	}
	var lastwindow, lastlength uint16
	for _, t := range bitmap {
		window := t / 256
		length := (t-window*256)/8 + 1
		if window > lastwindow && lastlength != 0 { // New window, jump to the new offset
			off += int(lastlength) + 2
			lastlength = 0
		}
		if window < lastwindow || length < lastlength {
			return len(msg), &Error{err: "nsec bits out of order"}
		}
		if off+2+int(length) > len(msg) {
			return len(msg), &Error{err: "overflow packing nsec"}
		}
		// Setting the window #
		msg[off] = byte(window)
		// Setting the octets length
		msg[off+1] = byte(length)
		// Setting the bit value for the type in the right octet
		msg[off+1+int(length)] |= byte(1 << (7 - t%8))
		lastwindow, lastlength = window, length
	}
	off += int(lastlength) + 2
	return off, nil
}

func unpackDataSVCB(msg []byte, off int) ([]SVCBKeyValue, int, error) {
	var xs []SVCBKeyValue
	var code uint16
	var length uint16
	var err error
	for off < len(msg) {
		code, off, err = unpackUint16(msg, off)
		if err != nil {
			return nil, len(msg), &Error{err: "overflow unpacking SVCB"}
		}
		length, off, err = unpackUint16(msg, off)
		if err != nil || off+int(length) > len(msg) {
			return nil, len(msg), &Error{err: "overflow unpacking SVCB"}
		}
		e := makeSVCBKeyValue(SVCBKey(code))
		if e == nil {
			return nil, len(msg), &Error{err: "bad SVCB key"}
		}
		if err := e.unpack(msg[off : off+int(length)]); err != nil {
			return nil, len(msg), err
		}
		if len(xs) > 0 && e.Key() <= xs[len(xs)-1].Key() {
			return nil, len(msg), &Error{err: "SVCB keys not in strictly increasing order"}
		}
		xs = append(xs, e)
		off += int(length)
	}
	return xs, off, nil
}

func packDataSVCB(pairs []SVCBKeyValue, msg []byte, off int) (int, error) {
	pairs = append([]SVCBKeyValue(nil), pairs...)
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].Key() < pairs[j].Key()
	})
	prev := svcb_RESERVED
	for _, el := range pairs {
		if el.Key() == prev {
			return len(msg), &Error{err: "repeated SVCB keys are not allowed"}
		}
		prev = el.Key()
		packed, err := el.pack()
		if err != nil {
			return len(msg), err
		}
		off, err = packUint16(uint16(el.Key()), msg, off)
		if err != nil {
			return len(msg), &Error{err: "overflow packing SVCB"}
		}
		off, err = packUint16(uint16(len(packed)), msg, off)
		if err != nil || off+len(packed) > len(msg) {
			return len(msg), &Error{err: "overflow packing SVCB"}
		}
		copy(msg[off:off+len(packed)], packed)
		off += len(packed)
	}
	return off, nil
}

func unpackDataDomainNames(msg []byte, off, end int) ([]string, int, error) {
	var (
		servers []string
		s       string
		err     error
	)
	if end > len(msg) {
		return nil, len(msg), &Error{err: "overflow unpacking domain names"}
	}
	for off < end {
		s, off, err = UnpackDomainName(msg, off)
		if err != nil {
			return servers, len(msg), err
		}
		servers = append(servers, s)
	}
	return servers, off, nil
}

func packDataDomainNames(names []string, msg []byte, off int, compression compressionMap, compress bool) (int, error) {
	var err error
	for _, name := range names {
		off, err = packDomainName(name, msg, off, compression, compress)
		if err != nil {
			return len(msg), err
		}
	}
	return off, nil
}

func packDataApl(data []APLPrefix, msg []byte, off int) (int, error) {
	var err error
	for i := range data {
		off, err = packDataAplPrefix(&data[i], msg, off)
		if err != nil {
			return len(msg), err
		}
	}
	return off, nil
}

func packDataAplPrefix(p *APLPrefix, msg []byte, off int) (int, error) {
	if len(p.Network.IP) != len(p.Network.Mask) {
		return len(msg), &Error{err: "address and mask lengths don't match"}
	}

	var err error
	prefix, _ := p.Network.Mask.Size()
	addr := p.Network.IP.Mask(p.Network.Mask)[:(prefix+7)/8]

	switch len(p.Network.IP) {
	case net.IPv4len:
		off, err = packUint16(1, msg, off)
	case net.IPv6len:
		off, err = packUint16(2, msg, off)
	default:
		err = &Error{err: "unrecognized address family"}
	}
	if err != nil {
		return len(msg), err
	}

	off, err = packUint8(uint8(prefix), msg, off)
	if err != nil {
		return len(msg), err
	}

	var n uint8
	if p.Negation {
		n = 0x80
	}

	// trim trailing zero bytes as specified in RFC3123 Sections 4.1 and 4.2.
	i := len(addr) - 1
	for ; i >= 0 && addr[i] == 0; i-- {
	}
	addr = addr[:i+1]

	adflen := uint8(len(addr)) & 0x7f
	off, err = packUint8(n|adflen, msg, off)
	if err != nil {
		return len(msg), err
	}

	if off+len(addr) > len(msg) {
		return len(msg), &Error{err: "overflow packing APL prefix"}
	}
	off += copy(msg[off:], addr)

	return off, nil
}

func unpackDataApl(msg []byte, off int) ([]APLPrefix, int, error) {
	var result []APLPrefix
	for off < len(msg) {
		prefix, end, err := unpackDataAplPrefix(msg, off)
		if err != nil {
			return nil, len(msg), err
		}
		off = end
		result = append(result, prefix)
	}
	return result, off, nil
}

func unpackDataAplPrefix(msg []byte, off int) (APLPrefix, int, error) {
	family, off, err := unpackUint16(msg, off)
	if err != nil {
		return APLPrefix{}, len(msg), &Error{err: "overflow unpacking APL prefix"}
	}
	prefix, off, err := unpackUint8(msg, off)
	if err != nil {
		return APLPrefix{}, len(msg), &Error{err: "overflow unpacking APL prefix"}
	}
	nlen, off, err := unpackUint8(msg, off)
	if err != nil {
		return APLPrefix{}, len(msg), &Error{err: "overflow unpacking APL prefix"}
	}

	var ip []byte
	switch family {
	case 1:
		ip = make([]byte, net.IPv4len)
	case 2:
		ip = make([]byte, net.IPv6len)
	default:
		return APLPrefix{}, len(msg), &Error{err: "unrecognized APL address family"}
	}
	if int(prefix) > 8*len(ip) {
		return APLPrefix{}, len(msg), &Error{err: "APL prefix too long"}
	}
	afdlen := int(nlen & 0x7f)
	if afdlen > len(ip) {
		return APLPrefix{}, len(msg), &Error{err: "APL length too long"}
	}
	if off+afdlen > len(msg) {
		return APLPrefix{}, len(msg), &Error{err: "overflow unpacking APL address"}
	}
	off += copy(ip, msg[off:off+afdlen])
	if afdlen > 0 {
		last := ip[afdlen-1]
		if last == 0 {
			return APLPrefix{}, len(msg), &Error{err: "extra APL address bits"}
		}
	}
	ipnet := net.IPNet{
		IP:   ip,
		Mask: net.CIDRMask(int(prefix), 8*len(ip)),
	}
	network := ipnet.IP.Mask(ipnet.Mask)
	if !network.Equal(ipnet.IP) {
		return APLPrefix{}, len(msg), &Error{err: "invalid APL address length"}
	}

	return APLPrefix{
		Negation: (nlen & 0x80) != 0,
		Network:  ipnet,
	}, off, nil
}
