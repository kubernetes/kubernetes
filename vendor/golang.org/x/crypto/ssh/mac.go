// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// Message authentication support

import (
	"crypto/fips140"
	"crypto/hmac"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"hash"
	"slices"
)

type macMode struct {
	keySize int
	etm     bool
	new     func(key []byte) hash.Hash
}

// truncatingMAC wraps around a hash.Hash and truncates the output digest to
// a given size.
type truncatingMAC struct {
	length int
	hmac   hash.Hash
}

func (t truncatingMAC) Write(data []byte) (int, error) {
	return t.hmac.Write(data)
}

func (t truncatingMAC) Sum(in []byte) []byte {
	out := t.hmac.Sum(in)
	return out[:len(in)+t.length]
}

func (t truncatingMAC) Reset() {
	t.hmac.Reset()
}

func (t truncatingMAC) Size() int {
	return t.length
}

func (t truncatingMAC) BlockSize() int { return t.hmac.BlockSize() }

// macModes defines the supported MACs. MACs not included are not supported
// and will not be negotiated, even if explicitly configured. When FIPS mode is
// enabled, only FIPS-approved algorithms are included.
var macModes = map[string]*macMode{}

func init() {
	macModes[HMACSHA512ETM] = &macMode{64, true, func(key []byte) hash.Hash {
		return hmac.New(sha512.New, key)
	}}
	macModes[HMACSHA256ETM] = &macMode{32, true, func(key []byte) hash.Hash {
		return hmac.New(sha256.New, key)
	}}
	macModes[HMACSHA512] = &macMode{64, false, func(key []byte) hash.Hash {
		return hmac.New(sha512.New, key)
	}}
	macModes[HMACSHA256] = &macMode{32, false, func(key []byte) hash.Hash {
		return hmac.New(sha256.New, key)
	}}

	if fips140.Enabled() {
		defaultMACs = slices.DeleteFunc(defaultMACs, func(algo string) bool {
			_, ok := macModes[algo]
			return !ok
		})
		return
	}

	macModes[HMACSHA1] = &macMode{20, false, func(key []byte) hash.Hash {
		return hmac.New(sha1.New, key)
	}}
	macModes[InsecureHMACSHA196] = &macMode{20, false, func(key []byte) hash.Hash {
		return truncatingMAC{12, hmac.New(sha1.New, key)}
	}}
}
