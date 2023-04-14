// Copyright (C) 2013-2018 by Maxim Bublis <b@codemonkey.ru>
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// Package uuid provides implementations of the Universally Unique Identifier
// (UUID), as specified in RFC-4122 and the Peabody RFC Draft (revision 03).
//
// RFC-4122[1] provides the specification for versions 1, 3, 4, and 5. The
// Peabody UUID RFC Draft[2] provides the specification for the new k-sortable
// UUIDs, versions 6 and 7.
//
// DCE 1.1[3] provides the specification for version 2, but version 2 support
// was removed from this package in v4 due to some concerns with the
// specification itself. Reading the spec, it seems that it would result in
// generating UUIDs that aren't very unique. In having read the spec it seemed
// that our implementation did not meet the spec. It also seems to be at-odds
// with RFC 4122, meaning we would need quite a bit of special code to support
// it. Lastly, there were no Version 2 implementations that we could find to
// ensure we were understanding the specification correctly.
//
// [1] https://tools.ietf.org/html/rfc4122
// [2] https://datatracker.ietf.org/doc/html/draft-peabody-dispatch-new-uuid-format-03
// [3] http://pubs.opengroup.org/onlinepubs/9696989899/chap5.htm#tagcjh_08_02_01_01
package uuid

import (
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"time"
)

// Size of a UUID in bytes.
const Size = 16

// UUID is an array type to represent the value of a UUID, as defined in RFC-4122.
type UUID [Size]byte

// UUID versions.
const (
	_  byte = iota
	V1      // Version 1 (date-time and MAC address)
	_       // Version 2 (date-time and MAC address, DCE security version) [removed]
	V3      // Version 3 (namespace name-based)
	V4      // Version 4 (random)
	V5      // Version 5 (namespace name-based)
	V6      // Version 6 (k-sortable timestamp and random data, field-compatible with v1) [peabody draft]
	V7      // Version 7 (k-sortable timestamp and random data) [peabody draft]
	_       // Version 8 (k-sortable timestamp, meant for custom implementations) [peabody draft] [not implemented]
)

// UUID layout variants.
const (
	VariantNCS byte = iota
	VariantRFC4122
	VariantMicrosoft
	VariantFuture
)

// UUID DCE domains.
const (
	DomainPerson = iota
	DomainGroup
	DomainOrg
)

// Timestamp is the count of 100-nanosecond intervals since 00:00:00.00,
// 15 October 1582 within a V1 UUID. This type has no meaning for other
// UUID versions since they don't have an embedded timestamp.
type Timestamp uint64

const _100nsPerSecond = 10000000

// Time returns the UTC time.Time representation of a Timestamp
func (t Timestamp) Time() (time.Time, error) {
	secs := uint64(t) / _100nsPerSecond
	nsecs := 100 * (uint64(t) % _100nsPerSecond)

	return time.Unix(int64(secs)-(epochStart/_100nsPerSecond), int64(nsecs)), nil
}

// TimestampFromV1 returns the Timestamp embedded within a V1 UUID.
// Returns an error if the UUID is any version other than 1.
func TimestampFromV1(u UUID) (Timestamp, error) {
	if u.Version() != 1 {
		err := fmt.Errorf("uuid: %s is version %d, not version 1", u, u.Version())
		return 0, err
	}

	low := binary.BigEndian.Uint32(u[0:4])
	mid := binary.BigEndian.Uint16(u[4:6])
	hi := binary.BigEndian.Uint16(u[6:8]) & 0xfff

	return Timestamp(uint64(low) + (uint64(mid) << 32) + (uint64(hi) << 48)), nil
}

// TimestampFromV6 returns the Timestamp embedded within a V6 UUID. This
// function returns an error if the UUID is any version other than 6.
//
// This is implemented based on revision 03 of the Peabody UUID draft, and may
// be subject to change pending further revisions. Until the final specification
// revision is finished, changes required to implement updates to the spec will
// not be considered a breaking change. They will happen as a minor version
// releases until the spec is final.
func TimestampFromV6(u UUID) (Timestamp, error) {
	if u.Version() != 6 {
		return 0, fmt.Errorf("uuid: %s is version %d, not version 6", u, u.Version())
	}

	hi := binary.BigEndian.Uint32(u[0:4])
	mid := binary.BigEndian.Uint16(u[4:6])
	low := binary.BigEndian.Uint16(u[6:8]) & 0xfff

	return Timestamp(uint64(low) + (uint64(mid) << 12) + (uint64(hi) << 28)), nil
}

// Nil is the nil UUID, as specified in RFC-4122, that has all 128 bits set to
// zero.
var Nil = UUID{}

// Predefined namespace UUIDs.
var (
	NamespaceDNS  = Must(FromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8"))
	NamespaceURL  = Must(FromString("6ba7b811-9dad-11d1-80b4-00c04fd430c8"))
	NamespaceOID  = Must(FromString("6ba7b812-9dad-11d1-80b4-00c04fd430c8"))
	NamespaceX500 = Must(FromString("6ba7b814-9dad-11d1-80b4-00c04fd430c8"))
)

// IsNil returns if the UUID is equal to the nil UUID
func (u UUID) IsNil() bool {
	return u == Nil
}

// Version returns the algorithm version used to generate the UUID.
func (u UUID) Version() byte {
	return u[6] >> 4
}

// Variant returns the UUID layout variant.
func (u UUID) Variant() byte {
	switch {
	case (u[8] >> 7) == 0x00:
		return VariantNCS
	case (u[8] >> 6) == 0x02:
		return VariantRFC4122
	case (u[8] >> 5) == 0x06:
		return VariantMicrosoft
	case (u[8] >> 5) == 0x07:
		fallthrough
	default:
		return VariantFuture
	}
}

// Bytes returns a byte slice representation of the UUID.
func (u UUID) Bytes() []byte {
	return u[:]
}

// encodeCanonical encodes the canonical RFC-4122 form of UUID u into the
// first 36 bytes dst.
func encodeCanonical(dst []byte, u UUID) {
	const hextable = "0123456789abcdef"
	dst[8] = '-'
	dst[13] = '-'
	dst[18] = '-'
	dst[23] = '-'
	for i, x := range [16]byte{
		0, 2, 4, 6,
		9, 11,
		14, 16,
		19, 21,
		24, 26, 28, 30, 32, 34,
	} {
		c := u[i]
		dst[x] = hextable[c>>4]
		dst[x+1] = hextable[c&0x0f]
	}
}

// String returns a canonical RFC-4122 string representation of the UUID:
// xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx.
func (u UUID) String() string {
	var buf [36]byte
	encodeCanonical(buf[:], u)
	return string(buf[:])
}

// Format implements fmt.Formatter for UUID values.
//
// The behavior is as follows:
// The 'x' and 'X' verbs output only the hex digits of the UUID, using a-f for 'x' and A-F for 'X'.
// The 'v', '+v', 's' and 'q' verbs return the canonical RFC-4122 string representation.
// The 'S' verb returns the RFC-4122 format, but with capital hex digits.
// The '#v' verb returns the "Go syntax" representation, which is a 16 byte array initializer.
// All other verbs not handled directly by the fmt package (like '%p') are unsupported and will return
// "%!verb(uuid.UUID=value)" as recommended by the fmt package.
func (u UUID) Format(f fmt.State, c rune) {
	if c == 'v' && f.Flag('#') {
		fmt.Fprintf(f, "%#v", [Size]byte(u))
		return
	}
	switch c {
	case 'x', 'X':
		b := make([]byte, 32)
		hex.Encode(b, u[:])
		if c == 'X' {
			toUpperHex(b)
		}
		_, _ = f.Write(b)
	case 'v', 's', 'S':
		b, _ := u.MarshalText()
		if c == 'S' {
			toUpperHex(b)
		}
		_, _ = f.Write(b)
	case 'q':
		b := make([]byte, 38)
		b[0] = '"'
		encodeCanonical(b[1:], u)
		b[37] = '"'
		_, _ = f.Write(b)
	default:
		// invalid/unsupported format verb
		fmt.Fprintf(f, "%%!%c(uuid.UUID=%s)", c, u.String())
	}
}

func toUpperHex(b []byte) {
	for i, c := range b {
		if 'a' <= c && c <= 'f' {
			b[i] = c - ('a' - 'A')
		}
	}
}

// SetVersion sets the version bits.
func (u *UUID) SetVersion(v byte) {
	u[6] = (u[6] & 0x0f) | (v << 4)
}

// SetVariant sets the variant bits.
func (u *UUID) SetVariant(v byte) {
	switch v {
	case VariantNCS:
		u[8] = (u[8]&(0xff>>1) | (0x00 << 7))
	case VariantRFC4122:
		u[8] = (u[8]&(0xff>>2) | (0x02 << 6))
	case VariantMicrosoft:
		u[8] = (u[8]&(0xff>>3) | (0x06 << 5))
	case VariantFuture:
		fallthrough
	default:
		u[8] = (u[8]&(0xff>>3) | (0x07 << 5))
	}
}

// Must is a helper that wraps a call to a function returning (UUID, error)
// and panics if the error is non-nil. It is intended for use in variable
// initializations such as
//
//	var packageUUID = uuid.Must(uuid.FromString("123e4567-e89b-12d3-a456-426655440000"))
func Must(u UUID, err error) UUID {
	if err != nil {
		panic(err)
	}
	return u
}
