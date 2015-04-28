// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package s2k implements the various OpenPGP string-to-key transforms as
// specified in RFC 4800 section 3.7.1.
package s2k

import (
	"crypto"
	"hash"
	"io"
	"strconv"

	"golang.org/x/crypto/openpgp/errors"
)

// Config collects configuration parameters for s2k key-stretching
// transformatioms. A nil *Config is valid and results in all default
// values. Currently, Config is used only by the Serialize function in
// this package.
type Config struct {
	// Hash is the default hash function to be used. If
	// nil, SHA1 is used.
	Hash crypto.Hash
	// S2KCount is only used for symmetric encryption. It
	// determines the strength of the passphrase stretching when
	// the said passphrase is hashed to produce a key. S2KCount
	// should be between 1024 and 65011712, inclusive. If Config
	// is nil or S2KCount is 0, the value 65536 used. Not all
	// values in the above range can be represented. S2KCount will
	// be rounded up to the next representable value if it cannot
	// be encoded exactly. When set, it is strongly encrouraged to
	// use a value that is at least 65536. See RFC 4880 Section
	// 3.7.1.3.
	S2KCount int
}

func (c *Config) hash() crypto.Hash {
	if c == nil || uint(c.Hash) == 0 {
		// SHA1 is the historical default in this package.
		return crypto.SHA1
	}

	return c.Hash
}

func (c *Config) encodedCount() uint8 {
	if c == nil || c.S2KCount == 0 {
		return 96 // The common case. Correspoding to 65536
	}

	i := c.S2KCount
	switch {
	// Behave like GPG. Should we make 65536 the lowest value used?
	case i < 1024:
		i = 1024
	case i > 65011712:
		i = 65011712
	}

	return encodeCount(i)
}

// encodeCount converts an iterative "count" in the range 1024 to
// 65011712, inclusive, to an encoded count. The return value is the
// octet that is actually stored in the GPG file. encodeCount panics
// if i is not in the above range (encodedCount above takes care to
// pass i in the correct range). See RFC 4880 Section 3.7.7.1.
func encodeCount(i int) uint8 {
	if i < 1024 || i > 65011712 {
		panic("count arg i outside the required range")
	}

	for encoded := 0; encoded < 256; encoded++ {
		count := decodeCount(uint8(encoded))
		if count >= i {
			return uint8(encoded)
		}
	}

	return 255
}

// decodeCount returns the s2k mode 3 iterative "count" corresponding to
// the encoded octet c.
func decodeCount(c uint8) int {
	return (16 + int(c&15)) << (uint32(c>>4) + 6)
}

// Simple writes to out the result of computing the Simple S2K function (RFC
// 4880, section 3.7.1.1) using the given hash and input passphrase.
func Simple(out []byte, h hash.Hash, in []byte) {
	Salted(out, h, in, nil)
}

var zero [1]byte

// Salted writes to out the result of computing the Salted S2K function (RFC
// 4880, section 3.7.1.2) using the given hash, input passphrase and salt.
func Salted(out []byte, h hash.Hash, in []byte, salt []byte) {
	done := 0
	var digest []byte

	for i := 0; done < len(out); i++ {
		h.Reset()
		for j := 0; j < i; j++ {
			h.Write(zero[:])
		}
		h.Write(salt)
		h.Write(in)
		digest = h.Sum(digest[:0])
		n := copy(out[done:], digest)
		done += n
	}
}

// Iterated writes to out the result of computing the Iterated and Salted S2K
// function (RFC 4880, section 3.7.1.3) using the given hash, input passphrase,
// salt and iteration count.
func Iterated(out []byte, h hash.Hash, in []byte, salt []byte, count int) {
	combined := make([]byte, len(in)+len(salt))
	copy(combined, salt)
	copy(combined[len(salt):], in)

	if count < len(combined) {
		count = len(combined)
	}

	done := 0
	var digest []byte
	for i := 0; done < len(out); i++ {
		h.Reset()
		for j := 0; j < i; j++ {
			h.Write(zero[:])
		}
		written := 0
		for written < count {
			if written+len(combined) > count {
				todo := count - written
				h.Write(combined[:todo])
				written = count
			} else {
				h.Write(combined)
				written += len(combined)
			}
		}
		digest = h.Sum(digest[:0])
		n := copy(out[done:], digest)
		done += n
	}
}

// Parse reads a binary specification for a string-to-key transformation from r
// and returns a function which performs that transform.
func Parse(r io.Reader) (f func(out, in []byte), err error) {
	var buf [9]byte

	_, err = io.ReadFull(r, buf[:2])
	if err != nil {
		return
	}

	hash, ok := HashIdToHash(buf[1])
	if !ok {
		return nil, errors.UnsupportedError("hash for S2K function: " + strconv.Itoa(int(buf[1])))
	}
	if !hash.Available() {
		return nil, errors.UnsupportedError("hash not available: " + strconv.Itoa(int(hash)))
	}
	h := hash.New()

	switch buf[0] {
	case 0:
		f := func(out, in []byte) {
			Simple(out, h, in)
		}
		return f, nil
	case 1:
		_, err = io.ReadFull(r, buf[:8])
		if err != nil {
			return
		}
		f := func(out, in []byte) {
			Salted(out, h, in, buf[:8])
		}
		return f, nil
	case 3:
		_, err = io.ReadFull(r, buf[:9])
		if err != nil {
			return
		}
		count := decodeCount(buf[8])
		f := func(out, in []byte) {
			Iterated(out, h, in, buf[:8], count)
		}
		return f, nil
	}

	return nil, errors.UnsupportedError("S2K function")
}

// Serialize salts and stretches the given passphrase and writes the
// resulting key into key. It also serializes an S2K descriptor to
// w. The key stretching can be configured with c, which may be
// nil. In that case, sensible defaults will be used.
func Serialize(w io.Writer, key []byte, rand io.Reader, passphrase []byte, c *Config) error {
	var buf [11]byte
	buf[0] = 3 /* iterated and salted */
	buf[1], _ = HashToHashId(c.hash())
	salt := buf[2:10]
	if _, err := io.ReadFull(rand, salt); err != nil {
		return err
	}
	encodedCount := c.encodedCount()
	count := decodeCount(encodedCount)
	buf[10] = encodedCount
	if _, err := w.Write(buf[:]); err != nil {
		return err
	}

	Iterated(key, c.hash().New(), passphrase, salt, count)
	return nil
}

// hashToHashIdMapping contains pairs relating OpenPGP's hash identifier with
// Go's crypto.Hash type. See RFC 4880, section 9.4.
var hashToHashIdMapping = []struct {
	id   byte
	hash crypto.Hash
	name string
}{
	{1, crypto.MD5, "MD5"},
	{2, crypto.SHA1, "SHA1"},
	{3, crypto.RIPEMD160, "RIPEMD160"},
	{8, crypto.SHA256, "SHA256"},
	{9, crypto.SHA384, "SHA384"},
	{10, crypto.SHA512, "SHA512"},
	{11, crypto.SHA224, "SHA224"},
}

// HashIdToHash returns a crypto.Hash which corresponds to the given OpenPGP
// hash id.
func HashIdToHash(id byte) (h crypto.Hash, ok bool) {
	for _, m := range hashToHashIdMapping {
		if m.id == id {
			return m.hash, true
		}
	}
	return 0, false
}

// HashIdToString returns the name of the hash function corresponding to the
// given OpenPGP hash id, or panics if id is unknown.
func HashIdToString(id byte) (name string, ok bool) {
	for _, m := range hashToHashIdMapping {
		if m.id == id {
			return m.name, true
		}
	}

	return "", false
}

// HashIdToHash returns an OpenPGP hash id which corresponds the given Hash.
func HashToHashId(h crypto.Hash) (id byte, ok bool) {
	for _, m := range hashToHashIdMapping {
		if m.hash == h {
			return m.id, true
		}
	}
	return 0, false
}
