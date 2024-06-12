// Copyright 2018 Google Inc.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uuid

import (
	"bytes"
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"strings"
	"sync"
)

// A UUID is a 128 bit (16 byte) Universal Unique IDentifier as defined in RFC
// 4122.
type UUID [16]byte

// A Version represents a UUID's version.
type Version byte

// A Variant represents a UUID's variant.
type Variant byte

// Constants returned by Variant.
const (
	Invalid   = Variant(iota) // Invalid UUID
	RFC4122                   // The variant specified in RFC4122
	Reserved                  // Reserved, NCS backward compatibility.
	Microsoft                 // Reserved, Microsoft Corporation backward compatibility.
	Future                    // Reserved for future definition.
)

const randPoolSize = 16 * 16

var (
	rander      = rand.Reader // random function
	poolEnabled = false
	poolMu      sync.Mutex
	poolPos     = randPoolSize     // protected with poolMu
	pool        [randPoolSize]byte // protected with poolMu
)

type invalidLengthError struct{ len int }

func (err invalidLengthError) Error() string {
	return fmt.Sprintf("invalid UUID length: %d", err.len)
}

// IsInvalidLengthError is matcher function for custom error invalidLengthError
func IsInvalidLengthError(err error) bool {
	_, ok := err.(invalidLengthError)
	return ok
}

// Parse decodes s into a UUID or returns an error if it cannot be parsed.  Both
// the standard UUID forms defined in RFC 4122
// (xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx and
// urn:uuid:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx) are decoded.  In addition,
// Parse accepts non-standard strings such as the raw hex encoding
// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx and 38 byte "Microsoft style" encodings,
// e.g.  {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}.  Only the middle 36 bytes are
// examined in the latter case.  Parse should not be used to validate strings as
// it parses non-standard encodings as indicated above.
func Parse(s string) (UUID, error) {
	var uuid UUID
	switch len(s) {
	// xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
	case 36:

	// urn:uuid:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
	case 36 + 9:
		if !strings.EqualFold(s[:9], "urn:uuid:") {
			return uuid, fmt.Errorf("invalid urn prefix: %q", s[:9])
		}
		s = s[9:]

	// {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}
	case 36 + 2:
		s = s[1:]

	// xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
	case 32:
		var ok bool
		for i := range uuid {
			uuid[i], ok = xtob(s[i*2], s[i*2+1])
			if !ok {
				return uuid, errors.New("invalid UUID format")
			}
		}
		return uuid, nil
	default:
		return uuid, invalidLengthError{len(s)}
	}
	// s is now at least 36 bytes long
	// it must be of the form  xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
	if s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-' {
		return uuid, errors.New("invalid UUID format")
	}
	for i, x := range [16]int{
		0, 2, 4, 6,
		9, 11,
		14, 16,
		19, 21,
		24, 26, 28, 30, 32, 34,
	} {
		v, ok := xtob(s[x], s[x+1])
		if !ok {
			return uuid, errors.New("invalid UUID format")
		}
		uuid[i] = v
	}
	return uuid, nil
}

// ParseBytes is like Parse, except it parses a byte slice instead of a string.
func ParseBytes(b []byte) (UUID, error) {
	var uuid UUID
	switch len(b) {
	case 36: // xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
	case 36 + 9: // urn:uuid:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
		if !bytes.EqualFold(b[:9], []byte("urn:uuid:")) {
			return uuid, fmt.Errorf("invalid urn prefix: %q", b[:9])
		}
		b = b[9:]
	case 36 + 2: // {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}
		b = b[1:]
	case 32: // xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
		var ok bool
		for i := 0; i < 32; i += 2 {
			uuid[i/2], ok = xtob(b[i], b[i+1])
			if !ok {
				return uuid, errors.New("invalid UUID format")
			}
		}
		return uuid, nil
	default:
		return uuid, invalidLengthError{len(b)}
	}
	// s is now at least 36 bytes long
	// it must be of the form  xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
	if b[8] != '-' || b[13] != '-' || b[18] != '-' || b[23] != '-' {
		return uuid, errors.New("invalid UUID format")
	}
	for i, x := range [16]int{
		0, 2, 4, 6,
		9, 11,
		14, 16,
		19, 21,
		24, 26, 28, 30, 32, 34,
	} {
		v, ok := xtob(b[x], b[x+1])
		if !ok {
			return uuid, errors.New("invalid UUID format")
		}
		uuid[i] = v
	}
	return uuid, nil
}

// MustParse is like Parse but panics if the string cannot be parsed.
// It simplifies safe initialization of global variables holding compiled UUIDs.
func MustParse(s string) UUID {
	uuid, err := Parse(s)
	if err != nil {
		panic(`uuid: Parse(` + s + `): ` + err.Error())
	}
	return uuid
}

// FromBytes creates a new UUID from a byte slice. Returns an error if the slice
// does not have a length of 16. The bytes are copied from the slice.
func FromBytes(b []byte) (uuid UUID, err error) {
	err = uuid.UnmarshalBinary(b)
	return uuid, err
}

// Must returns uuid if err is nil and panics otherwise.
func Must(uuid UUID, err error) UUID {
	if err != nil {
		panic(err)
	}
	return uuid
}

// Validate returns an error if s is not a properly formatted UUID in one of the following formats:
//   xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
//   urn:uuid:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
//   xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
//   {xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx}
// It returns an error if the format is invalid, otherwise nil.
func Validate(s string) error {
	switch len(s) {
	// Standard UUID format
	case 36:

	// UUID with "urn:uuid:" prefix
	case 36 + 9:
		if !strings.EqualFold(s[:9], "urn:uuid:") {
			return fmt.Errorf("invalid urn prefix: %q", s[:9])
		}
		s = s[9:]

	// UUID enclosed in braces
	case 36 + 2:
		if s[0] != '{' || s[len(s)-1] != '}' {
			return fmt.Errorf("invalid bracketed UUID format")
		}
		s = s[1 : len(s)-1]

	// UUID without hyphens
	case 32:
		for i := 0; i < len(s); i += 2 {
			_, ok := xtob(s[i], s[i+1])
			if !ok {
				return errors.New("invalid UUID format")
			}
		}

	default:
		return invalidLengthError{len(s)}
	}

	// Check for standard UUID format
	if len(s) == 36 {
		if s[8] != '-' || s[13] != '-' || s[18] != '-' || s[23] != '-' {
			return errors.New("invalid UUID format")
		}
		for _, x := range []int{0, 2, 4, 6, 9, 11, 14, 16, 19, 21, 24, 26, 28, 30, 32, 34} {
			if _, ok := xtob(s[x], s[x+1]); !ok {
				return errors.New("invalid UUID format")
			}
		}
	}

	return nil
}

// String returns the string form of uuid, xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
// , or "" if uuid is invalid.
func (uuid UUID) String() string {
	var buf [36]byte
	encodeHex(buf[:], uuid)
	return string(buf[:])
}

// URN returns the RFC 2141 URN form of uuid,
// urn:uuid:xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx,  or "" if uuid is invalid.
func (uuid UUID) URN() string {
	var buf [36 + 9]byte
	copy(buf[:], "urn:uuid:")
	encodeHex(buf[9:], uuid)
	return string(buf[:])
}

func encodeHex(dst []byte, uuid UUID) {
	hex.Encode(dst, uuid[:4])
	dst[8] = '-'
	hex.Encode(dst[9:13], uuid[4:6])
	dst[13] = '-'
	hex.Encode(dst[14:18], uuid[6:8])
	dst[18] = '-'
	hex.Encode(dst[19:23], uuid[8:10])
	dst[23] = '-'
	hex.Encode(dst[24:], uuid[10:])
}

// Variant returns the variant encoded in uuid.
func (uuid UUID) Variant() Variant {
	switch {
	case (uuid[8] & 0xc0) == 0x80:
		return RFC4122
	case (uuid[8] & 0xe0) == 0xc0:
		return Microsoft
	case (uuid[8] & 0xe0) == 0xe0:
		return Future
	default:
		return Reserved
	}
}

// Version returns the version of uuid.
func (uuid UUID) Version() Version {
	return Version(uuid[6] >> 4)
}

func (v Version) String() string {
	if v > 15 {
		return fmt.Sprintf("BAD_VERSION_%d", v)
	}
	return fmt.Sprintf("VERSION_%d", v)
}

func (v Variant) String() string {
	switch v {
	case RFC4122:
		return "RFC4122"
	case Reserved:
		return "Reserved"
	case Microsoft:
		return "Microsoft"
	case Future:
		return "Future"
	case Invalid:
		return "Invalid"
	}
	return fmt.Sprintf("BadVariant%d", int(v))
}

// SetRand sets the random number generator to r, which implements io.Reader.
// If r.Read returns an error when the package requests random data then
// a panic will be issued.
//
// Calling SetRand with nil sets the random number generator to the default
// generator.
func SetRand(r io.Reader) {
	if r == nil {
		rander = rand.Reader
		return
	}
	rander = r
}

// EnableRandPool enables internal randomness pool used for Random
// (Version 4) UUID generation. The pool contains random bytes read from
// the random number generator on demand in batches. Enabling the pool
// may improve the UUID generation throughput significantly.
//
// Since the pool is stored on the Go heap, this feature may be a bad fit
// for security sensitive applications.
//
// Both EnableRandPool and DisableRandPool are not thread-safe and should
// only be called when there is no possibility that New or any other
// UUID Version 4 generation function will be called concurrently.
func EnableRandPool() {
	poolEnabled = true
}

// DisableRandPool disables the randomness pool if it was previously
// enabled with EnableRandPool.
//
// Both EnableRandPool and DisableRandPool are not thread-safe and should
// only be called when there is no possibility that New or any other
// UUID Version 4 generation function will be called concurrently.
func DisableRandPool() {
	poolEnabled = false
	defer poolMu.Unlock()
	poolMu.Lock()
	poolPos = randPoolSize
}

// UUIDs is a slice of UUID types.
type UUIDs []UUID

// Strings returns a string slice containing the string form of each UUID in uuids.
func (uuids UUIDs) Strings() []string {
	var uuidStrs = make([]string, len(uuids))
	for i, uuid := range uuids {
		uuidStrs[i] = uuid.String()
	}
	return uuidStrs
}
