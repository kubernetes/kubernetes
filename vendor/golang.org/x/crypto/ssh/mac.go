// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssh

// Message authentication support

import (
	"crypto/hmac"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"hash"
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

var macModes = map[string]*macMode{
	HMACSHA512ETM: {64, true, func(key []byte) hash.Hash {
		return hmac.New(sha512.New, key)
	}},
	HMACSHA256ETM: {32, true, func(key []byte) hash.Hash {
		return hmac.New(sha256.New, key)
	}},
	HMACSHA512: {64, false, func(key []byte) hash.Hash {
		return hmac.New(sha512.New, key)
	}},
	HMACSHA256: {32, false, func(key []byte) hash.Hash {
		return hmac.New(sha256.New, key)
	}},
	HMACSHA1: {20, false, func(key []byte) hash.Hash {
		return hmac.New(sha1.New, key)
	}},
	InsecureHMACSHA196: {20, false, func(key []byte) hash.Hash {
		return truncatingMAC{12, hmac.New(sha1.New, key)}
	}},
}
